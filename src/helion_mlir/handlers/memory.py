from __future__ import annotations

import helion.language.atomic_ops as hl_atomic_ops
import helion.language.memory_ops as hl_memory_ops

from ..custom_op import set_memory_space


def register_memory_handlers(direct: dict[object, str], predicates: list[tuple[object, str]]) -> None:
    direct[hl_memory_ops.load] = "visit_load"
    direct[set_memory_space] = "visit_set_memory_space"
    direct[hl_memory_ops.store] = "visit_store"
    direct[hl_atomic_ops.atomic_add] = "visit_atomic_add"
