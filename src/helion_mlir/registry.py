from __future__ import annotations

import importlib
from collections.abc import Callable

from .handlers import (
    register_compute_handlers,
    register_control_flow_handlers,
    register_memory_handlers,
    register_symbol_handlers,
    register_tensor_handlers,
    register_tile_handlers,
)


def build_handler_registry() -> tuple[dict[object, str], list[tuple[Callable[[object], bool], str]]]:
    """Build default handler registry for IRVisitor dispatch."""
    direct: dict[object, str] = {}
    predicates: list[tuple[Callable[[object], bool], str]] = []
    register_symbol_handlers(direct, predicates)
    register_tile_handlers(direct, predicates)
    register_control_flow_handlers(direct, predicates)
    register_memory_handlers(direct, predicates)
    register_tensor_handlers(direct, predicates)
    register_compute_handlers(direct, predicates)
    return direct, predicates


def load_custom_ops() -> dict[str, tuple[object, ...]]:
    """Lazily load supported custom ops.

    Loads the packaged helion-mlir custom-op namespace.
    """
    module_candidates = [
        "helion_mlir.custom_op",
    ]

    loaded: dict[str, list[object]] = {"gather": [], "broadcast": []}
    for module_name in module_candidates:
        try:
            module = importlib.import_module(module_name)
            gather = getattr(module, "gather", None)
            broadcast = getattr(module, "broadcast", None)
            if gather is not None:
                loaded["gather"].append(gather)
            if broadcast is not None:
                loaded["broadcast"].append(broadcast)
        except Exception:
            continue

    return {k: tuple(v) for k, v in loaded.items() if v}
