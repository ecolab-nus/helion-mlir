from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Collection, Iterator, Mapping

from .mlir_utils import MLIROutputHelper
from .models import KernelAnalysis, LoopScope


@dataclass
class GraphScope:
    node_values: dict[str, str]
    node_types: dict[str, str]
    loop_result_values: dict[str, Any]
    range_index_block_ids: dict[str, int]


class LoweringSession:
    """Mutable lowering state layered on top of immutable kernel analysis."""

    def __init__(self, analysis: KernelAnalysis):
        self.analysis = analysis
        self.bound_kernel = analysis.bound_kernel
        self.mlir_output_helper = MLIROutputHelper()
        self.host_tensors: dict[str, str] = {}
        self.node_values: dict[str, str] = {}
        self.node_types: dict[str, str] = {}
        self.initial_acc_ssa: dict[str, str] = {}
        self.graphs: dict[int, Any] = dict(analysis.graph_inventory.inner_graphs)
        self.block_size_ssa: dict[int, str] = {}
        self.reduction_trip_counts: dict[int, str] = {}
        self.loop_result_values: dict[str, Any] = {}
        self.gather_dim_overrides: dict[str, dict[int, str]] = {}
        self.range_index_block_ids: dict[str, int] = {}
        self.loop_scopes: list[LoopScope] = []
        self.loop_iter_args: dict[str, str] = {}
        self.current_loop_result: str | list[str] | None = None
        self.current_block_id: int | None = None
        from .resolvers import LoopResolver, SymbolResolver, TypeResolver

        self.symbol_resolver = SymbolResolver(self)
        self.type_resolver = TypeResolver(self, self.symbol_resolver)
        self.loop_resolver = LoopResolver(self)

    @property
    def kernel_name(self) -> str:
        return self.analysis.kernel_name

    @property
    def arg_mlir_types(self) -> dict[str, str]:
        return self.analysis.host_tensors.arg_types

    @property
    def host_tensor_types(self) -> dict[str, str]:
        return self.analysis.host_tensors.tensor_types

    @property
    def env(self) -> Any:
        return self.bound_kernel.env

    @property
    def device_ir(self) -> Any:
        return self.bound_kernel.host_function.device_ir

    @property
    def all_grid_block_ids(self) -> list[list[int]]:
        return [list(g) for g in self.device_ir.grid_block_ids]

    @property
    def assume_divisible_tiles(self) -> bool:
        return self.analysis.assume_divisible_tiles

    def tile_is_divisible(self, block_id: int) -> bool:
        canonical_id = self.resolve_block_id(block_id)
        return (
            self.analysis.assume_divisible_tiles
            or canonical_id in self.analysis.divisible_block_ids
        )

    def resolve_block_id(self, block_id: int) -> int:
        return self.analysis.block_info.canonical_aliases.get(block_id, block_id)

    def get_loop_extent(self, block_id: int) -> int | None:
        return self.analysis.block_info.loop_extents.get(block_id)

    def get_natural_upper_bound(self, block_id: int) -> int | None:
        return self.analysis.block_info.natural_upper_bounds.get(block_id)

    def compute_mlir_type_from_fake_tensor(self, fake_tensor: Any) -> str:
        return self.type_resolver.compute_mlir_type_from_fake_tensor(fake_tensor)

    def compute_mlir_memref_type_from_fake_tensor(self, fake_tensor: Any) -> str:
        return self.type_resolver.compute_mlir_memref_type_from_fake_tensor(fake_tensor)

    def bind_node_value(self, name: str, ssa: str, *, node_type: str | None = None) -> str:
        self.node_values[name] = ssa
        if node_type is not None:
            self.node_types[name] = node_type
        return ssa

    def lookup_node_value(self, name: str, default: str | None = None) -> str | None:
        return self.node_values.get(name, default)

    def record_loop_result(self, name: str, value: Any, *, count: int | None = None) -> None:
        self.loop_result_values[name] = value
        if count is not None:
            self.loop_result_values["_count"] = count

    def lookup_loop_result_projection(self, source_name: str, index: int) -> str | None:
        source_ssa = self.node_values.get(source_name)
        if source_ssa is None or source_name not in self.loop_result_values:
            return None
        return f"{source_ssa}#{index}"

    @contextmanager
    def push_graph_scope(self) -> Iterator[None]:
        snapshot = GraphScope(
            node_values=self.node_values.copy(),
            node_types=self.node_types.copy(),
            loop_result_values=self.loop_result_values.copy(),
            range_index_block_ids=self.range_index_block_ids.copy(),
        )
        try:
            yield
        finally:
            self.node_values = snapshot.node_values
            self.node_types = snapshot.node_types
            self.loop_result_values = snapshot.loop_result_values
            self.range_index_block_ids = snapshot.range_index_block_ids

    @contextmanager
    def push_loop_scope(
        self,
        *,
        block_id: int,
        iter_args: dict[str, str] | None = None,
        bounds: dict[int, tuple[str, str]] | None = None,
    ) -> Iterator[LoopScope]:
        scope = LoopScope(
            block_id=block_id,
            iter_args=dict(iter_args or {}),
            bounds=dict(bounds or {}),
            current_result=self.current_loop_result,
        )
        self.loop_scopes.append(scope)
        old_loop_iter_args = self.loop_iter_args
        old_current_block_id = self.current_block_id
        self.loop_iter_args = dict(self.loop_iter_args)
        self.loop_iter_args.update(scope.iter_args)
        self.current_block_id = block_id
        try:
            yield scope
        finally:
            self.loop_iter_args = old_loop_iter_args
            self.current_block_id = old_current_block_id
            self.loop_scopes.pop()

    def active_loop_bounds(self) -> dict[int, tuple[str, str]]:
        combined: dict[int, tuple[str, str]] = {}
        for scope in self.loop_scopes:
            combined.update(scope.bounds)
        return combined


class LoweringContext(LoweringSession):
    """Compatibility shim for older imports."""

    def __init__(
        self,
        bound_kernel: Any,
        assume_divisible_tiles: bool = False,
        divisible_tiles: Collection[str | int] = (),
        tile_upper_bounds: Mapping[str | int, int] | None = None,
    ):
        from .analysis import build_kernel_analysis

        super().__init__(
            build_kernel_analysis(
                bound_kernel,
                assume_divisible_tiles=assume_divisible_tiles,
                divisible_tiles=divisible_tiles,
                tile_upper_bounds=tile_upper_bounds,
            )
        )


def collect_reduction_block_ids(device_ir: Any) -> list[int]:
    block_ids: list[int] = []
    for graph_info in device_ir.graphs:
        candidate = getattr(graph_info, "block_ids", None)
        if candidate:
            for block_id in candidate:
                if block_id not in block_ids:
                    block_ids.append(block_id)
    return block_ids
