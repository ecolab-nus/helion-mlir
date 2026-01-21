"""MLIR emission from Helion Device IR.

This module generates MLIR from Helion Device IR by walking FX graph nodes
instruction-by-instruction. Each Device IR operation is mapped to a
corresponding MLIR operation:

- _get_symnode -> SSA lookup via Origin (BlockSizeOrigin -> block_size_ssa)
- full -> tensor.empty + linalg.fill
- _for_loop -> affine.for + recursive visit
- _phi -> Loop result SSA (simplified merge pattern detection)
- _host_tensor -> function argument mapping
- aten.sym_size.int -> inline concrete value or tensor.dim
- load -> tensor.extract_slice
- store -> tensor.insert_slice

Architecture:
- IRVisitor: Walks FX graphs and generates MLIR via handlers
- LoweringContext: Holds state during lowering
- MLIROutputHelper: Handles MLIR text emission and SSA naming
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

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

from .debug_utils import run_dce_cleanup



def generate_mlir(
    bound_kernel: "BoundKernel",
    *,
    cleanup: bool = True,
) -> str:
    """Generate MLIR by walking Device IR instruction-by-instruction.
    
    This function visits Device IR FX nodes sequentially, generating
    corresponding MLIR operations with 1:1 correspondence.
    
    Args:
        bound_kernel: A bound Helion kernel with fake_args set
        cleanup: Whether to run mlir-opt canonicalize/cse passes (default: True), you should keep it to False if you want more intuitive variable names.
    
    Returns:
        MLIR text representation of the kernel
    """
    from helion._compiler.device_ir import RootGraphInfo, ForLoopGraphInfo
    
    # Create lowering context from bound kernel
    ctx = LoweringContext(bound_kernel)
    builder = ctx.mlir_output_helper
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
    
    # Emit block size lookups for ALL blocks
    # - Concrete int sizes -> emit arith.constant
    # - Symbolic sizes -> emit loom.get_symbol
    block_size_ssa = {}
    for info in ctx.env.block_sizes:
        block_id = info.block_id
        ssa = f"%block_size_{block_id}"
        
        if isinstance(info.size, int):
            # Concrete size -> no need to emit anything
            pass
        else:
            # Symbolic size -> emit loom.get_symbol
            builder.emit(
                f'{ssa} = "loom.get_symbol"() '
                f'{{name = "block_size_{block_id}"}} : () -> index'
            )
        block_size_ssa[block_id] = ssa

    
    # Pre-compute trip counts for reduction loops (needed for affine.for trip count in IRVisitor)
    # Uses affine.apply with affine_map for consistency with affine.parallel loops
    for_trip_counts = {}
    
    # Iterate over reduction loops
    reduction_block_ids = collect_reduction_block_ids(device_ir)
    parallel_block_ids_set = set(ctx.parallel_block_ids)
    reduction_block_ids = [bid for bid in reduction_block_ids if bid not in parallel_block_ids_set]
    
    for block_id in reduction_block_ids:
        info = ctx.env.block_sizes[block_id]
        total_extent = ctx.loop_extents[block_id]  # Pre-computed extent
        
        trip_count_ssa = builder.fresh("trip_count")
        
        if isinstance(info.size, int):
            # Concrete tile size - compute trip count at compile time
            tile_size = info.size
            trip_count = max(1, math.ceil(total_extent / tile_size))
            builder.emit(f'{trip_count_ssa} = arith.constant {trip_count} : index')
        else:
            # Symbolic tile size - use affine.apply to compute trip count
            # Trip count = ceil(total_extent / tile_size)
            tile_size_ssa = block_size_ssa[block_id]
            builder.emit(
                f'{trip_count_ssa} = affine.apply '
                f'affine_map<()[s0] -> ({total_extent} ceildiv s0)>()[{tile_size_ssa}]'
            )
        
        # Store for use in visit_for_loop
        for_trip_counts[block_id] = trip_count_ssa
    
    # Make block sizes and trip counts available to context
    ctx.block_size_ssa = block_size_ssa
    ctx.reduction_trip_counts = for_trip_counts
    
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
                 
            iv_names.append(f"%iv_block_{block_id}")
        
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
    
    mlir_text = builder.build()
    
    if cleanup:
        try:
            mlir_text = run_dce_cleanup(mlir_text)
        except (FileNotFoundError, RuntimeError):
            # If mlir-opt is not available or fails, return raw text
            pass
    
    return mlir_text

