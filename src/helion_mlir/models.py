from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GraphInventory:
    root_ids: tuple[int, ...]
    root_graphs: dict[int, Any]
    inner_graphs: dict[int, Any]
    reachable_inner_ids: frozenset[int]


@dataclass(frozen=True)
class BlockInfoSummary:
    canonical_aliases: dict[int, int]
    loop_extents: dict[int, int]
    used_block_ids: frozenset[int]
    used_canonical_block_ids: frozenset[int]


@dataclass(frozen=True)
class HostTensorInfo:
    tensor_types: dict[str, str]
    arg_types: dict[str, str]
    ordered_tensor_names: tuple[str, ...] = field(default_factory=tuple)
    stored_tensor_names: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class LoopTripInfo:
    block_id: int
    trip_count_ssa: str | None = None
    lower_bound_ssa: str | None = None
    upper_bound_ssa: str | None = None


@dataclass(frozen=True)
class KernelAnalysis:
    bound_kernel: Any
    kernel_name: str
    graph_inventory: GraphInventory
    block_info: BlockInfoSummary
    host_tensors: HostTensorInfo
    module_attributes: dict[str, tuple[object, str]]
    reduction_block_ids: tuple[int, ...]
    assume_divisible_tiles: bool = False


@dataclass
class LoopScope:
    block_id: int
    iter_args: dict[str, str] = field(default_factory=dict)
    bounds: dict[int, tuple[str, str]] = field(default_factory=dict)
    current_result: str | list[str] | None = None


@dataclass(frozen=True)
class SubviewSpec:
    memref_type: str
    result_type: str
    subview_type: str
    offsets: tuple[tuple[str, bool], ...]
    sizes: tuple[tuple[str, bool], ...]
    strides: tuple[str, ...]
    retained_dim_positions: tuple[int, ...]
