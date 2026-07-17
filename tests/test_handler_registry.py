from __future__ import annotations

import operator

from torch.ops import aten

import helion.language._tracing_ops as hl_tracing_ops
import helion.language.creation_ops as hl_creation_ops
import helion.language.memory_ops as hl_memory_ops
import helion.language.tile_ops as hl_tile_ops
import helion.language.view_ops as hl_view_ops

from helion_mlir.custom_op import set_memory_space

from helion_mlir.handlers import (
    register_compute_handlers,
    register_control_flow_handlers,
    register_memory_handlers,
    register_symbol_handlers,
    register_tensor_handlers,
    register_tile_handlers,
)


def test_handler_registry_covers_core_targets() -> None:
    direct: dict[object, str] = {}
    predicates: list[tuple[object, str]] = []

    register_symbol_handlers(direct, predicates)
    register_tile_handlers(direct, predicates)
    register_control_flow_handlers(direct, predicates)
    register_memory_handlers(direct, predicates)
    register_tensor_handlers(direct, predicates)
    register_compute_handlers(direct, predicates)

    assert direct[hl_tracing_ops._get_symnode] == "visit_get_symnode"
    assert direct[hl_creation_ops.full] == "visit_full"
    assert direct[hl_tracing_ops._for_loop] == "visit_for_loop"
    assert direct[hl_memory_ops.load] == "visit_load"
    assert direct[set_memory_space] == "visit_set_memory_space"
    assert direct[hl_tile_ops.tile_begin] == "visit_tile_begin"
    assert direct[hl_view_ops.subscript] == "visit_subscript"
    assert direct[operator.getitem] == "visit_getitem"
    assert any(handler == "visit_aten_compute" for _, handler in predicates)
    assert any(handler == "visit_dot" for _, handler in predicates)


def test_handler_registry_covers_aten_sym_size_and_full() -> None:
    direct: dict[object, str] = {}
    predicates: list[tuple[object, str]] = []

    register_symbol_handlers(direct, predicates)
    register_tensor_handlers(direct, predicates)

    assert direct[aten.sym_size.int] == "visit_sym_size"
    assert direct[aten.full.default] == "visit_aten_full"
