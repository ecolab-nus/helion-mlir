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

    Supports multi-grid kernels separated by ``hl.barrier()``.  Each grid
    group produces its own ``affine.parallel`` region.  Block IDs that tile
    the same source dimension across grids are *aliased* so they share the
    same ``loom.block_size_*`` symbol and IV names.

    Args:
        bound_kernel: A bound Helion kernel with fake_args set
        cleanup: Whether to run mlir-opt canonicalize/cse passes (default: True),
                 you should keep it to False if you want more intuitive variable
                 names.

    Returns:
        MLIR text representation of the kernel
    """
    from helion._compiler.device_ir import (
        RootGraphInfo,
        ForLoopGraphInfo,
        ReductionLoopGraphInfo,
    )

    # ------------------------------------------------------------------
    # 1. Create lowering context
    # ------------------------------------------------------------------
    ctx = LoweringContext(bound_kernel)
    builder = ctx.mlir_output_helper
    device_ir = bound_kernel.host_function.device_ir

    # ------------------------------------------------------------------
    # 2. Classify graphs (filter out rolled reductions)
    # ------------------------------------------------------------------
    rolled_ids = {
        info.new_graph_id
        for info in device_ir.rolled_reductions
        if info.new_graph_id is not None
    }

    root_graphs: dict[int, RootGraphInfo] = {}
    inner_graphs: dict[int, ForLoopGraphInfo] = {}

    for graph_info in device_ir.graphs:
        if graph_info.graph_id in rolled_ids:
            continue
        if isinstance(graph_info, RootGraphInfo):
            root_graphs[graph_info.graph_id] = graph_info
        elif isinstance(graph_info, (ForLoopGraphInfo, ReductionLoopGraphInfo)):
            inner_graphs[graph_info.graph_id] = graph_info

    if not root_graphs:
        raise ValueError("No RootGraphInfo found in device_ir.graphs")

    # Map root_ids[i] → grid_block_ids[i]
    root_ids = list(device_ir.root_ids)
    grid_groups = ctx.all_grid_block_ids
    alias = ctx.block_id_alias

    # ------------------------------------------------------------------
    # 3. Module header (block-size attributes)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 4. Function signature
    # ------------------------------------------------------------------
    func_args = []
    for tensor_name, tensor_type in ctx.host_tensor_types.items():
        ssa_name = f"%{tensor_name}"
        func_args.append((ssa_name, tensor_type))
        ctx.host_tensors[tensor_name] = ssa_name

    args_str = ", ".join(f"{n}: {t}" for n, t in func_args)
    builder.emit(f"func.func @{ctx.kernel_name}({args_str}) {{")
    builder.push()

    # ------------------------------------------------------------------
    # 5. Emit block-size SSA values (canonical only)
    # ------------------------------------------------------------------
    block_size_ssa: dict[int, str] = {}
    emitted_canonical: set[int] = set()

    for info in ctx.env.block_sizes:
        canonical_id = alias.get(info.block_id, info.block_id)
        if canonical_id in emitted_canonical:
            block_size_ssa[info.block_id] = block_size_ssa[canonical_id]
            continue
        emitted_canonical.add(canonical_id)

        sym_name = next(iter(info.debug_names), f"block_{canonical_id}")
        ssa = f"%{sym_name}"

        if not isinstance(info.size, int):
            upper_bound = ctx.loop_extents[info.block_id]
            is_reduction = str(info.reduction).lower()
            builder.emit(
                f'{ssa} = "loom.sym"() {{'
                f'symbol_ref = @{sym_name}, '
                f'upper_bound = {upper_bound} : index, '
                f'is_reduction = {is_reduction}'
                f'}} : () -> index'
            )
        block_size_ssa[canonical_id] = ssa
        block_size_ssa[info.block_id] = ssa

    ctx.block_size_ssa = block_size_ssa

    # ------------------------------------------------------------------
    # 6. Register all inner (for-loop / reduction-loop) graphs
    # ------------------------------------------------------------------
    visitor = IRVisitor(ctx)
    for gid, ginfo in inner_graphs.items():
        ctx.graphs[gid] = ginfo

    # ------------------------------------------------------------------
    # 7. Pre-compute reduction trip counts
    # ------------------------------------------------------------------
    all_parallel_block_ids: set[int] = set()
    for grp in grid_groups:
        all_parallel_block_ids.update(grp)

    reduction_block_ids = collect_reduction_block_ids(device_ir)
    reduction_block_ids = [
        bid for bid in reduction_block_ids if bid not in all_parallel_block_ids
    ]

    for_trip_counts: dict[int, str] = {}
    for block_id in reduction_block_ids:
        canonical_id = ctx.resolve_block_id(block_id)
        info = ctx.env.block_sizes[block_id]
        total_extent = ctx.loop_extents.get(block_id)
        if total_extent is None:
            continue

        trip_count_ssa = builder.fresh("trip_count")
        if isinstance(info.size, int):
            trip_count = max(1, math.ceil(total_extent / info.size))
            builder.emit(f'{trip_count_ssa} = arith.constant {trip_count} : index')
        else:
            tile_size_ssa = block_size_ssa[canonical_id]
            builder.emit(
                f'{trip_count_ssa} = affine.apply '
                f'affine_map<()[s0] -> ({total_extent} ceildiv s0)>()[{tile_size_ssa}]'
            )
        for_trip_counts[block_id] = trip_count_ssa

    ctx.reduction_trip_counts = for_trip_counts

    # ------------------------------------------------------------------
    # 8. Emit one affine.parallel per grid group
    # ------------------------------------------------------------------
    for grid_idx, grid_block_ids in enumerate(grid_groups):
        root_gid = root_ids[grid_idx]
        root_graph = root_graphs.get(root_gid)
        if root_graph is None:
            raise ValueError(
                f"Root graph {root_gid} for grid group {grid_idx} not found"
            )

        # Compute trip counts for this grid's parallel dimensions
        ub_ssas: list[str] = []
        iv_names: list[str] = []

        for block_id in grid_block_ids:
            canonical_id = ctx.resolve_block_id(block_id)
            info = ctx.env.block_sizes[block_id]
            total_extent = ctx.loop_extents[block_id]

            if isinstance(info.size, int):
                val = math.ceil(total_extent / info.size)
                ssa = builder.fresh("trip_count")
                builder.emit(f'{ssa} = arith.constant {val} : index')
                ub_ssas.append(ssa)
            else:
                size_ssa = block_size_ssa[canonical_id]
                trip_ssa = builder.fresh("trip_count")
                builder.emit(
                    f'{trip_ssa} = affine.apply '
                    f'affine_map<()[s0] -> ({total_extent} ceildiv s0)>()[{size_ssa}]'
                )
                ub_ssas.append(trip_ssa)

            iv_names.append(f"%iv_block_{canonical_id}")

        lb_str = "(" + ", ".join(["0"] * len(iv_names)) + ")"
        ub_str = "(" + ", ".join(ub_ssas) + ")"
        iv_str = ", ".join(iv_names)

        builder.emit(f"affine.parallel ({iv_str}) = {lb_str} to {ub_str} {{")
        builder.push()

        # Reset per-graph mutable state so graphs don't bleed into each other
        ctx.node_values = {}
        ctx.node_types = {}
        ctx.initial_acc_ssa = {}
        ctx.loop_result_values = {}

        visitor.visit_graph(root_graph)

        builder.emit("affine.yield")
        builder.pop()
        builder.emit("}")

    # ------------------------------------------------------------------
    # 9. Close function / module
    # ------------------------------------------------------------------
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
            pass

    return mlir_text

