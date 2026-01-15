"""MLIR emission from Helion Device IR.

This module generates MLIR from Helion Device IR by walking FX graph nodes
instruction-by-instruction. Each Device IR operation is mapped to a
corresponding MLIR operation:

- _get_symnode -> loom.get_symbol
- full -> tensor.empty + linalg.fill
- _for_loop -> affine.for + recursive visit
- _phi -> helion.phi
- _host_tensor -> function argument mapping
- aten.sym_size.int -> inline concrete value
- load -> tensor.extract_slice
- store -> tensor.insert_slice

Architecture:
- IRVisitor: Walks FX graphs and generates MLIR via handlers
- LoweringContext: Holds state during lowering
- MLIRBuilder: Handles MLIR text emission and SSA naming
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable, TYPE_CHECKING

import math

from .mlir_builder import (
    format_tensor_type,
    format_shape_attr,
)
from .lowering_context import (
    LoweringContext,
    first_debug_name,
    collect_reduction_block_ids,
)
from .ir_visitor import IRVisitor

if TYPE_CHECKING:
    from helion._compiler.device_ir import RootGraphInfo, ForLoopGraphInfo

REPO_ROOT = Path(__file__).resolve().parents[2]
HELION_OPT_CANDIDATES = [
    REPO_ROOT / "build" / "mlir" / "helion-opt",
    REPO_ROOT / "build" / "bin" / "helion-opt",
    Path("/mnt/fast/llvm-mlir/bin/helion-opt"),
    Path("/mnt/fast/llvm-mlir/bin/mlir-opt"),
]


def _collect_host_tensor_names(device_ir, rolled_ids: set[int]) -> list[str]:
    """Collect all unique host tensor names from _host_tensor calls across all graphs.
    
    This pre-scans all FX graphs to identify every tensor that will be referenced
    via _host_tensor calls, so they can all be registered as function parameters.
    
    Args:
        device_ir: The DeviceIR containing all graphs
        rolled_ids: Set of graph IDs to skip (rolled reductions)
    
    Returns:
        List of unique host tensor names in order of first occurrence
    """
    import helion.language._tracing_ops as hl_tracing_ops
    
    host_tensor_names: list[str] = []
    seen: set[str] = set()
    
    for graph_info in device_ir.graphs:
        if graph_info.graph_id in rolled_ids:
            continue
        for node in graph_info.graph.nodes:
            if node.op == "call_function" and node.target is hl_tracing_ops._host_tensor:
                name = node.args[0]
                if name not in seen:
                    host_tensor_names.append(name)
                    seen.add(name)
    
    return host_tensor_names


def generate_mlir(
    bound_kernel: "BoundKernel",
    *,
    kernel_name: str | None = None,
) -> str:
    """Generate MLIR by walking Device IR instruction-by-instruction.
    
    This function visits Device IR FX nodes sequentially, generating
    corresponding MLIR operations with 1:1 correspondence.
    
    Args:
        bound_kernel: A bound Helion kernel with fake_args set
        kernel_name: Optional name override for the generated MLIR function.
                     If not provided, uses the kernel function's __name__.
    
    Returns:
        MLIR text representation of the kernel
    """
    from helion._compiler.device_ir import RootGraphInfo, ForLoopGraphInfo
    
    # Create lowering context from bound kernel
    ctx = LoweringContext(bound_kernel)
    builder = ctx.builder
    device_ir = bound_kernel.host_function.device_ir
    
    # Use provided name or derive from kernel
    actual_kernel_name = kernel_name if kernel_name else ctx.kernel_name
    
    # Find root and for-loop graphs
    root_graph = None
    for_loop_graphs: dict[int, ForLoopGraphInfo] = {}
    
    # Filter out rolled reduction graphs
    rolled_ids = {
        info.new_graph_id 
        for info in device_ir.rolled_reductions 
        if info.new_graph_id is not None
    }
    
    for graph_info in device_ir.graphs:
        if graph_info.graph_id in rolled_ids:
            continue
        if isinstance(graph_info, RootGraphInfo):
            root_graph = graph_info
        elif isinstance(graph_info, ForLoopGraphInfo):
            for_loop_graphs[graph_info.graph_id] = graph_info
    
    if root_graph is None:
        raise ValueError("No RootGraphInfo found in device_ir.graphs")
    
    # Validate we have exactly one root and one for_loop graph for now
    if len(for_loop_graphs) > 1:
        raise ValueError(
            f"Expected at most 1 ForLoopGraphInfo, found {len(for_loop_graphs)}. "
            "Multiple nested loops not yet supported."
        )
    
    # Collect ALL host tensor names from _host_tensor calls across ALL graphs
    # This ensures every tensor referenced via _host_tensor becomes a function parameter
    # ONLY tensors in this list should be in the function signature
    all_host_tensor_names = _collect_host_tensor_names(device_ir, rolled_ids)
    
    # Build mapping from tensor name to type info
    kernel_arg_by_name = {arg.name: arg for arg in ctx.kernel_args if arg.is_tensor}
    
    # Emit module start with tile size attributes
    module_attrs = ctx.get_module_attributes()
    builder.emit_module_start(module_attrs)
    
    # Build function signature ONLY from host tensors actually used in Device IR
    # This ensures parameter names match what the Device IR expects
    func_args = []
    for tensor_name in all_host_tensor_names:
        ssa_name = f"%{tensor_name}"
        
        if tensor_name == 'out':
            # Output tensor - use dynamic type like other _host_tensor parameters
            # The 'out' tensor comes from _host_tensor('out') and should be treated
            # the same as input tensors (dynamic shape)
            tensor_type = f"tensor<?x?xf32>"
        elif tensor_name in kernel_arg_by_name:
            # This is a kernel arg that's also used in Device IR
            tensor_type = kernel_arg_by_name[tensor_name].mlir_type or f"tensor<?x?xf32>"
        else:
            # Derived tensor (view) - use dynamic tensor type
            tensor_type = f"tensor<?x?xf32>"
        
        func_args.append((ssa_name, tensor_type))
        # Pre-register in host_tensors for later lookup
        ctx.host_tensors[tensor_name] = ssa_name
    
    # Function returns void (output is via out parameter)
    result_type = None  # void return
    
    # Emit function start with void return
    builder.emit_func_start(actual_kernel_name, func_args, result_type)
    
    # Create visitor and register all graphs in context
    visitor = IRVisitor(ctx)
    for graph_id, graph_info in for_loop_graphs.items():
        ctx.graphs[graph_id] = graph_info
    
    # Emit block size lookups for ALL blocks using loom.get_symbol
    block_size_ssa = {}
    for info in ctx.env.block_sizes:
        block_id = info.block_id
        ssa = builder.fresh(f"block_size_{block_id}")
        builder.emit(
            f'{ssa} = "loom.get_symbol"() '
            f'{{name = "block_size_{block_id}"}} : () -> index'
        )
        block_size_ssa[block_id] = ssa
    
    # Pre-compute trip counts for reduction loops (needed for affine.for trip count in IRVisitor)
    # These are usually constant or symbolic, but here we emit them as constants if possible
    # or rely on the IRVisitor to compute them if they are complex.
    reduction_trip_counts = {}
    
    # Iterate over reduction loops
    reduction_block_ids = collect_reduction_block_ids(device_ir)
    parallel_block_ids_set = set(ctx.parallel_block_ids)
    reduction_block_ids = [bid for bid in reduction_block_ids if bid not in parallel_block_ids_set]
    
    for block_id in reduction_block_ids:
        # Assuming block_id is the list index as per allocate_block_size implementation.
        info = ctx.env.block_sizes[block_id]
        block_name = first_debug_name(info.debug_names, fallback=f"block_{block_id}")
        
        # Calculate trip count
        total_extent = ctx.loop_extents[block_name]  # Pre-computed extent
        
        trip_count_ssa = builder.fresh("trip_count")
        
        if isinstance(info.size, int):
            tile_size = info.size
            trip_count = max(1, math.ceil(total_extent / tile_size))
            builder.emit(f'{trip_count_ssa} = arith.constant {trip_count} : index')
            
        else:
            # Symbolic trip count logic
            # Trip count = ceil(total_extent / tile_size)
            # tile_size is symbolic -> get from block_size_ssa
            
            # Block size SSA must exist since we emitted all of them above
            tile_size_ssa = block_size_ssa[block_id]
            
            # Generate calculation: (total_extent + tile_size - 1) // tile_size
            # Since total_extent is constant here, we can emit it
            
            # %total = arith.constant ...
            total_ssa = builder.fresh("total_extent")
            builder.emit(f'{total_ssa} = arith.constant {total_extent} : index')
            
            # %c1 = arith.constant 1
            c1_ssa = builder.fresh("c1")
            builder.emit(f'{c1_ssa} = arith.constant 1 : index')
            
            # %t_minus_1 = sub total, 1
            t_minus_1 = builder.fresh("t_minus_1")
            builder.emit(f'{t_minus_1} = arith.subi {total_ssa}, {c1_ssa} : index')
            
            # %num = add t_minus_1, tile_size
            num_ssa = builder.fresh("numerator")
            builder.emit(f'{num_ssa} = arith.addi {t_minus_1}, {tile_size_ssa} : index')
            
            # %trip_count = divui num, tile_size
            builder.emit(f'{trip_count_ssa} = arith.divui {num_ssa}, {tile_size_ssa} : index')
        
        # Store in visitor for use in visit_for_loop
        reduction_trip_counts[block_id] = trip_count_ssa
    
    # Make block sizes and trip counts available to context
    ctx.block_size_ssa = block_size_ssa
    ctx.reduction_trip_counts = reduction_trip_counts
    
    # Emit affine.parallel for grid blocks
    if ctx.parallel_block_ids:
        # We have parallel loops (e.g., M, N tiling)
        # Emit nested affine.parallel
        
        # Collect loop bounds
        lower_bounds = []
        upper_bounds = []
        steps = []
        iv_names = []
        
        for block_id in ctx.parallel_block_ids:
            info = ctx.env.block_sizes[block_id]
            block_name = first_debug_name(info.debug_names, fallback=f"block_{block_id}")
            total_extent = ctx.loop_extents[block_name]
            
            # 0 to total_extent (M/N) with step tile_size
            lower_bounds.append("0")
            upper_bounds.append(str(total_extent))
            
            if isinstance(info.size, int):
                 steps.append(str(info.size))
            else:
                 steps.append("1") # fallback step for symbolic (or symbolic constant if we had it)
                 
            iv_names.append(f"%iv_{block_name}")
        
        # Format parallel loop
        lb_str = "(" + ", ".join(lower_bounds) + ")"
        ub_str = "(" + ", ".join(upper_bounds) + ")"
        step_str = "(" + ", ".join(steps) + ")"
        iv_str = ", ".join(iv_names)
        
        builder.emit(
            f"affine.parallel ({iv_str}) = {lb_str} to {ub_str} step {step_str} {{"
        )
        builder.push()
        
        # Inside the parallel loop, we need to map the IVs (which are loop indices e.g. 0, 128, 256)
        # to the block indices expected by the code (if necessary).
        # The current logic assumes iv_blockX IS the loop index (offset).
        
        # Visit the root graph
        visitor.visit_graph(root_graph)
        
        builder.emit("affine.yield")
        builder.pop()
        builder.emit("}")
    
    # Emit function end (void return - no return statement needed)
    builder.emit("return")
    builder.emit_func_end()
    builder.emit_module_end()
    
    return builder.build()

# -----------------------------------------------------------------------------
# Validation utility
# -----------------------------------------------------------------------------


def validate_with_helion_opt(
    mlir_text: str,
    *,
    opt_path: str | Path | None = None,
    extra_args: Iterable[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run `helion-opt` (falling back to `mlir-opt`) to confirm the emitted IR parses."""
    tool_candidates: Iterable[Path] = HELION_OPT_CANDIDATES if opt_path is None else [Path(opt_path)]
    
    tool: Path | None = None
    for candidate in tool_candidates:
        if candidate.exists():
            tool = candidate
            break
    
    if tool is None:
        raise FileNotFoundError(
            "Unable to locate `helion-opt` or `mlir-opt`. "
            "Pass `mlir_opt_path` explicitly once the project is built."
        )
    
    args = [str(tool)]
    if tool.name == "mlir-opt":
        args.append("-allow-unregistered-dialect")
    if extra_args:
        args.extend(extra_args)
    
    return subprocess.run(
        args,
        input=mlir_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
