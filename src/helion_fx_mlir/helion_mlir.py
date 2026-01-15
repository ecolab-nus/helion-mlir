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

from .mlir_builder import (
    format_tensor_type,
    format_shape_attr,
)
from .lowering_context import LoweringContext
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
    kernel_name: str = "helion_kernel",
) -> str:
    """Generate MLIR by walking Device IR instruction-by-instruction.
    
    This function visits Device IR FX nodes sequentially, generating
    corresponding MLIR operations with 1:1 correspondence.
    
    Args:
        bound_kernel: A bound Helion kernel with fake_args set
        kernel_name: Name for the generated MLIR function
    
    Returns:
        MLIR text representation of the kernel
    """
    from helion._compiler.device_ir import RootGraphInfo, ForLoopGraphInfo
    
    # Create lowering context from bound kernel
    ctx = LoweringContext.from_bound_kernel(bound_kernel, kernel_name)
    builder = ctx.builder
    device_ir = bound_kernel.host_function.device_ir
    
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
            tensor_type = ctx.tensor_type
        elif tensor_name in kernel_arg_by_name:
            # This is a kernel arg that's also used in Device IR
            tensor_type = kernel_arg_by_name[tensor_name].mlir_type or ctx.tensor_type
        else:
            # Derived tensor (view) - use dynamic tensor type
            tensor_type = ctx.tensor_type
        
        func_args.append((ssa_name, tensor_type))
        # Pre-register in host_tensors for later lookup
        ctx.host_tensors[tensor_name] = ssa_name
    
    # Function returns void (output is via out parameter)
    result_type = None  # void return
    
    # Emit function start with void return
    builder.emit_func_start(kernel_name, func_args, result_type)
    
    # Create visitor and register all graphs
    visitor = IRVisitor(ctx)
    for graph_id, graph_info in for_loop_graphs.items():
        visitor.register_graph(graph_id, graph_info)
    
    # Get grid block IDs for parallel loops
    grid_block_ids = device_ir.grid_block_ids
    
    # Emit block size lookups for grid blocks using loom.get_symbol
    block_size_ssa = {}
    for block_group in grid_block_ids:
        for block_id in block_group:
            ssa = builder.fresh(f"block_size_{block_id}")
            builder.emit(
                f'{ssa} = "loom.get_symbol"() '
                f'{{name = "block_size_{block_id}"}} : () -> index'
            )
            block_size_ssa[block_id] = ssa
    
    # Emit block size lookups for reduction loops (must be outside affine.parallel)
    reduction_trip_counts = {}
    for loop in ctx.reduction_loops:
        block_id = loop.block_id
        if block_id not in block_size_ssa:
            ssa = builder.fresh(f"block_size_{block_id}")
            builder.emit(
                f'{ssa} = "loom.get_symbol"() '
                f'{{name = "block_size_{block_id}"}} : () -> index'
            )
            block_size_ssa[block_id] = ssa
        else :
            raise ValueError(f"Block ID {block_id} not found in block_size_ssa")
        
        # Compute trip count for reduction loop
        if isinstance(loop.total_extent, int):
            tc = builder.fresh("apply")
            builder.emit(
                f'{tc} = affine.apply affine_map<()[s0] -> ({loop.total_extent} ceildiv s0)>()[{block_size_ssa[block_id]}]'
            )
            reduction_trip_counts[block_id] = tc
    
    # Make block sizes and trip counts available to visitor
    visitor.block_size_ssa = block_size_ssa
    visitor.reduction_trip_counts = reduction_trip_counts
    
    # Emit affine.parallel for grid blocks
    if grid_block_ids and grid_block_ids[0]:
        block_group = grid_block_ids[0]  # Assume single grid for now
        
        # Compute trip counts for each grid dimension
        trip_count_ssa = []
        for i, block_id in enumerate(block_group):
            loop = ctx.outer_loops[i] if i < len(ctx.outer_loops) else None
            if loop and isinstance(loop.total_extent, int):
                extent = loop.total_extent
                tc = builder.fresh("apply")
                block_ssa = block_size_ssa[block_id]
                builder.emit(
                    f'{tc} = affine.apply affine_map<()[s0] -> ({extent} ceildiv s0)>()[{block_ssa}]'
                )
                trip_count_ssa.append(tc)
            else:
                trip_count_ssa.append(block_size_ssa.get(block_id, "%unknown"))
        
        # Build affine.parallel header
        ivs = ", ".join([f"%iv_block{bid}" for bid in block_group])
        zeros = ", ".join(["0"] * len(block_group))
        ubs = ", ".join(trip_count_ssa)
        steps = ", ".join(["1"] * len(block_group))
        
        builder.emit(f"affine.parallel ({ivs}) = ({zeros}) to ({ubs}) step ({steps}) {{")
        builder.push()
        
        # Make block sizes and IVs available to visitor
        for block_id in block_group:
            visitor.node_values[f"block_size_{block_id}"] = block_size_ssa[block_id]
            # Register IV as a symbol for the visitor
            ctx.symbols[f"block_size_{block_id}"] = block_size_ssa[block_id]
    
    # Visit the root graph
    visitor.visit_graph(root_graph)
    
    # Close affine.parallel if opened
    if grid_block_ids and grid_block_ids[0]:
        builder.emit("affine.yield")
        builder.pop()
        builder.emit("}")
    
    # Emit function end (void return - no return statement needed)
    builder.emit("return")
    builder.emit_func_end()
    builder.emit_module_end()
    
    return builder.build()


# def _infer_output_shape(ctx: LoweringContext) -> list[int | None]:
#     """Infer output shape from outer loop extents.
    
#     The outer (parallel) loops define the output tensor dimensions.
#     For matmul: [tile_m, tile_n] -> [M, N]
#     """
#     shape = []
#     for loop in ctx.outer_loops:
#         extent = loop.total_extent
#         if isinstance(extent, int):
#             shape.append(extent)
#         else:
#             shape.append(None)  # Dynamic dimension
    
#     # Ensure we have at least 2 dimensions for typical 2D output tensors
#     while len(shape) < 2:
#         shape.append(None)
    
#     return shape


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
