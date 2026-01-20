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
- MLIROutputHelper: Handles MLIR text emission and SSA naming
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable, TYPE_CHECKING

import math

from .mlir_utils import (
    format_tensor_type,
    format_shape_attr,
)
from .lowering_context import (
    LoweringContext,
    collect_reduction_block_ids,
)
from .ir_visitor import IRVisitor

if TYPE_CHECKING:
    from helion._compiler.device_ir import RootGraphInfo, ForLoopGraphInfo

MLIR_OPT_CANDIDATES = [
    Path("/mnt/fast/llvm-mlir/bin/mlir-opt"),
    Path("/usr/bin/mlir-opt"),
    Path("/usr/local/bin/mlir-opt"),
]



def generate_mlir(
    bound_kernel: "BoundKernel"
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
    
    # Emit module start with tile size attributes
    module_attrs = ctx.get_module_attributes()
    if module_attrs:
        attr_strs = []
        for name, (value, typ) in module_attrs.items():
            if typ:
                attr_strs.append(f"{name} = {value} : {typ}")
            else:
                attr_strs.append(f"{name} = {value}")
        builder.emit(f"module attributes {{{', '.join(attr_strs)}}} {{")
    else:
        builder.emit("module {")
    builder.push()
    
    # Build function signature using pre-computed host tensor types from LoweringContext
    # host_tensor_types is computed during LoweringContext init by scanning all graphs
    func_args = []
    for tensor_name, tensor_type in ctx.host_tensor_types.items():
        ssa_name = f"%{tensor_name}"
        func_args.append((ssa_name, tensor_type))
        # Pre-register in host_tensors for later lookup
        ctx.host_tensors[tensor_name] = ssa_name
    
    # Function returns void (output is via out parameter)
    result_type = None  # void return
    
    # Emit function start with void return
    args_str = ", ".join(f"{arg_name}: {arg_type}" for arg_name, arg_type in func_args)
    if result_type:
        builder.emit(f"func.func @{ctx.kernel_name}({args_str}) -> {result_type} {{")
    else:
        builder.emit(f"func.func @{ctx.kernel_name}({args_str}) {{")
    builder.push()
    
    # Create visitor and register all graphs in context
    visitor = IRVisitor(ctx)
    for graph_id, graph_info in for_loop_graphs.items():
        ctx.graphs[graph_id] = graph_info
    
    # Emit block size lookups for ALL blocks using loom.get_symbol
    block_size_ssa = {}
    for info in ctx.env.block_sizes:
        block_id = info.block_id
        ssa = f"%block_size_{block_id}"
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
        
        # Calculate trip count
        total_extent = ctx.loop_extents[block_id]  # Pre-computed extent
        
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
            
            # Generate calculation: ceil(total_extent / tile_size)
            # Since total_extent is constant here, we can emit it
            
            # %total = arith.constant ...
            total_ssa = builder.fresh("total_extent")
            builder.emit(f'{total_ssa} = arith.constant {total_extent} : index')
            
            # %trip_count = ceildivui total, tile_size
            builder.emit(f'{trip_count_ssa} = arith.ceildivui {total_ssa}, {tile_size_ssa} : index')
        
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
        ub_ssas = []
        iv_names = []
        
        for block_id in ctx.parallel_block_ids:
            info = ctx.env.block_sizes[block_id]
            total_extent = ctx.loop_extents[block_id]
            
            # 0 to ceil(total / tile_size)
            if isinstance(info.size, int):
                val = math.ceil(total_extent / info.size)
                ssa = builder.fresh("trip_count")
                builder.emit(f'{ssa} = arith.constant {val} : index')
                ub_ssas.append(ssa)
            else:
                # Symbolic: ceil_div(total, tile_size)
                size_ssa = ctx.block_size_ssa[block_id]
                trip_ssa = builder.fresh("trip_count")
                
                # Use affine.apply to compute trip count with map
                builder.emit(
                    f'{trip_ssa} = affine.apply '
                    f'affine_map<()[s0] -> ({total_extent} ceildiv s0)>()[{size_ssa}]'
                )
                ub_ssas.append(trip_ssa)
                 
            iv_names.append(f"%iv_block{block_id}")
        
        # Format parallel loop
        # Lower bound: (0, 0, ...)
        lb_str = "(" + ", ".join(["0"] * len(iv_names)) + ")"
        # Upper bound: (%trip0, %trip1, ...)
        ub_str = "(" + ", ".join(ub_ssas) + ")"
        iv_str = ", ".join(iv_names)
        
        builder.emit(
            f"affine.parallel ({iv_str}) = {lb_str} to {ub_str} {{"
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
    builder.pop()
    builder.emit("}")
    builder.pop()
    builder.emit("}")
    
    return builder.build()

# -----------------------------------------------------------------------------
# Validation utility
# -----------------------------------------------------------------------------


def validate_with_mlir_opt(
    mlir_text: str,
    *,
    opt_path: str | Path | None = None,
    extra_args: Iterable[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run `mlir-opt` to confirm the emitted IR parses.
    
    Uses -allow-unregistered-dialect to allow loom.* and torch.* operations.
    """
    tool_candidates: Iterable[Path] = MLIR_OPT_CANDIDATES if opt_path is None else [Path(opt_path)]
    
    tool: Path | None = None
    for candidate in tool_candidates:
        if candidate.exists():
            tool = candidate
            break
    
    if tool is None:
        raise FileNotFoundError(
            "Unable to locate `mlir-opt`. "
            "Install LLVM/MLIR or pass `opt_path` explicitly."
        )
    
    args = [str(tool), "-allow-unregistered-dialect"]
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
