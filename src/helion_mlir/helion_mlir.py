"""MLIR emission from Helion Device IR.

The lowering pipeline is intentionally split into explicit stages:
- `build_kernel_analysis()` gathers immutable facts from `bound_kernel`
- `LoweringSession` owns mutable state during lowering
- `ModuleEmitter` owns module/function scaffolding and pre-emitted symbols
- `IRVisitor` walks FX graphs and lowers operations into MLIR text
"""

from __future__ import annotations

import math
from typing import Collection, Mapping, TYPE_CHECKING

from .analysis import build_kernel_analysis
from .emitter import ModuleEmitter
from .errors import MissingGraphError
from .ir_visitor import IRVisitor
from .session import LoweringSession

if TYPE_CHECKING:
    from helion._compiler.runtime import BoundKernel


def generate_mlir(
    bound_kernel: "BoundKernel",
    *,
    cleanup: bool = True,
    assume_divisible: bool = False,
    assume_divisible_tiles: bool = False,
    divisible_tiles: Collection[str | int] = (),
    tile_divisible: Mapping[str | int, bool] | None = None,
    tile_upper_bounds: Mapping[str | int, int] | None = None,
) -> str:
    """Lower a bound Helion kernel to MLIR.

    ``assume_divisible`` marks all tile loops as having no partial final tile,
    suppressing generated boundary checks. ``assume_divisible_tiles`` is kept
    as a compatibility alias.
    ``divisible_tiles`` marks individual Helion tiles as having no partial
    final tile. Entries may be tile variable names (preferred) or block ids.
    ``tile_divisible`` sets the ``asure_divisible`` bool attribute on emitted
    tile symbols. Entries may be tile variable names (preferred) or block ids.
    ``tile_upper_bounds`` overrides individual Helion tile upper bounds.
    Entries may be tile variable names (preferred) or block ids.
    """
    analysis = build_kernel_analysis(
        bound_kernel,
        assume_divisible=assume_divisible or assume_divisible_tiles,
        divisible_tiles=divisible_tiles,
        tile_divisible=tile_divisible,
        tile_upper_bounds=tile_upper_bounds,
    )
    session = LoweringSession(analysis)
    emitter = ModuleEmitter(session)
    emitter.emit_module_prelude()
    emitter.emit_block_size_symbols()
    emitter.emit_reduction_trip_counts()

    visitor = IRVisitor(session)
    root_ids = analysis.graph_inventory.root_ids
    root_graphs = analysis.graph_inventory.root_graphs

    for grid_idx, grid_block_ids in enumerate(session.all_grid_block_ids):
        root_gid = root_ids[grid_idx]
        root_graph = root_graphs.get(root_gid)
        if root_graph is None:
            raise MissingGraphError(f"Root graph {root_gid} for grid group {grid_idx} not found")

        ub_ssas: list[str] = []
        iv_names: list[str] = []
        for block_id in grid_block_ids:
            canonical_id = session.resolve_block_id(block_id)
            info = session.env.block_sizes[block_id]
            total_extent = session.get_loop_extent(block_id)
            if total_extent is None:
                raise ValueError(f"Missing loop extent for block_id={block_id}")
            if isinstance(info.size, int):
                val = math.ceil(total_extent / info.size)
                ssa = session.mlir_output_helper.fresh("trip_count")
                session.mlir_output_helper.emit(f"{ssa} = arith.constant {val} : index")
                ub_ssas.append(ssa)
            else:
                size_ssa = session.block_size_ssa[canonical_id]
                total_extent_ssa = session.mlir_output_helper.fresh("loop_extent")
                session.mlir_output_helper.emit(
                    f"{total_extent_ssa} = arith.constant {total_extent} : index"
                )
                trip_ssa = session.mlir_output_helper.fresh("trip_count")
                session.mlir_output_helper.emit(
                    f"{trip_ssa} = arith.ceildivui {total_extent_ssa}, {size_ssa} : index"
                )
                ub_ssas.append(trip_ssa)
            iv_names.append(f"%iv_block_{canonical_id}")

        lb_str = "(" + ", ".join(["0"] * len(iv_names)) + ")"
        ub_str = "(" + ", ".join(ub_ssas) + ")"
        iv_str = ", ".join(iv_names)
        session.mlir_output_helper.emit(f"affine.parallel ({iv_str}) = {lb_str} to {ub_str} {{")
        session.mlir_output_helper.push()

        session.node_values = {}
        session.node_types = {}
        session.initial_acc_ssa = {}
        session.loop_result_values = {}

        visitor.visit_graph(root_graph)

        session.mlir_output_helper.emit("affine.yield")
        session.mlir_output_helper.pop()
        session.mlir_output_helper.emit("}")

    return emitter.close_module(cleanup=cleanup)
