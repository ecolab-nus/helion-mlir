from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from .mlir_utils import format_memref_type, format_tensor_type, torch_dtype_to_mlir_element_type
from .models import BlockInfoSummary, GraphInventory, HostTensorInfo, KernelAnalysis


def _direct_host_tensor_name(node: Any, host_tensor_target: Any) -> str | None:
    if (
        getattr(node, "op", None) != "call_function"
        or getattr(node, "target", None) is not host_tensor_target
        or not getattr(node, "args", None)
    ):
        return None
    name = node.args[0]
    return name if isinstance(name, str) else None


def _collect_stored_tensor_names(
    graph_infos: list[Any],
    rolled_ids: set[int],
    *,
    host_tensor_target: Any,
    store_target: Any,
) -> tuple[str, ...]:
    stored_names: list[str] = []
    seen_stored_names: set[str] = set()

    for graph_info in graph_infos:
        if graph_info.graph_id in rolled_ids:
            continue
        for node in graph_info.graph.nodes:
            if (
                getattr(node, "op", None) != "call_function"
                or getattr(node, "target", None) is not store_target
            ):
                continue
            if not getattr(node, "args", None):
                continue
            tensor_node = node.args[0]
            name = _direct_host_tensor_name(tensor_node, host_tensor_target)
            if name is None or name in seen_stored_names:
                continue
            seen_stored_names.add(name)
            stored_names.append(name)

    return tuple(stored_names)


def _order_host_tensor_names(
    host_tensor_types: dict[str, str],
    arg_types: dict[str, str],
    stored_tensor_names: tuple[str, ...],
) -> tuple[str, ...]:
    stored = {name for name in stored_tensor_names if name in host_tensor_types}
    ordered_names: list[str] = []

    for name in arg_types:
        if name in host_tensor_types and name not in stored:
            ordered_names.append(name)

    for name in host_tensor_types:
        if name not in stored and name not in ordered_names:
            ordered_names.append(name)

    for name in stored_tensor_names:
        if name in host_tensor_types and name not in ordered_names:
            ordered_names.append(name)

    return tuple(ordered_names)


def build_kernel_analysis(
    bound_kernel: Any,
    *,
    assume_divisible_tiles: bool = False,
) -> KernelAnalysis:
    from helion._compiler.device_ir import IfGraphInfo, ReductionLoopGraphInfo, RootGraphInfo, ForLoopGraphInfo
    import helion.language._tracing_ops as hl_tracing_ops
    import helion.language.memory_ops as hl_memory_ops
    from helion._compiler.variable_origin import BlockSizeOrigin

    device_ir = bound_kernel.host_function.device_ir
    rolled_ids = {
        info.new_graph_id
        for info in device_ir.rolled_reductions
        if info.new_graph_id is not None
    }

    root_graphs: dict[int, RootGraphInfo] = {}
    inner_graphs: dict[int, Any] = {}
    for graph_info in device_ir.graphs:
        if graph_info.graph_id in rolled_ids:
            continue
        if isinstance(graph_info, RootGraphInfo):
            root_graphs[graph_info.graph_id] = graph_info
        elif isinstance(graph_info, (ForLoopGraphInfo, ReductionLoopGraphInfo, IfGraphInfo)):
            inner_graphs[graph_info.graph_id] = graph_info

    if not root_graphs:
        raise ValueError("No RootGraphInfo found in device_ir.graphs")

    root_ids = tuple(device_ir.root_ids)
    graph_lookup = {**root_graphs, **inner_graphs}
    reachable_inner_ids: set[int] = set()
    worklist: list[int] = [rid for rid in root_ids if rid in root_graphs]
    visited_graphs: set[int] = set()

    while worklist:
        gid = worklist.pop()
        if gid in visited_graphs:
            continue
        visited_graphs.add(gid)
        ginfo = graph_lookup.get(gid)
        if ginfo is None:
            continue
        for node in ginfo.graph.nodes:
            if node.op != "call_function":
                continue
            if node.target is hl_tracing_ops._for_loop:
                nested_gid = node.args[0]
            elif node.target is hl_tracing_ops._if:
                nested_gid = node.args[1]
            else:
                continue
            if isinstance(nested_gid, int) and nested_gid in inner_graphs:
                reachable_inner_ids.add(nested_gid)
                worklist.append(nested_gid)

    inner_graphs = {gid: g for gid, g in inner_graphs.items() if gid in reachable_inner_ids}

    param_names = list(bound_kernel.kernel.signature.parameters.keys())
    arg_types: dict[str, str] = {}
    for name, fake_arg in zip(param_names, bound_kernel.fake_args):
        if isinstance(fake_arg, torch.Tensor):
            shape = [
                int(s) if not isinstance(s, torch.SymInt) else None
                for s in fake_arg.shape
            ]
            element_type = torch_dtype_to_mlir_element_type(fake_arg.dtype)
            arg_types[name] = format_tensor_type(shape, element_type)
    if not arg_types:
        raise ValueError("No tensor arguments found")

    shape_env = bound_kernel.env.shape_env
    loop_extents: dict[int, int] = {}
    for info in bound_kernel.env.block_sizes:
        size = info.size
        extent: int | None = None
        if isinstance(size, int):
            extent = int(size)
        elif hasattr(size, "_sympy_"):
            sym = size._sympy_()
            if sym in shape_env.var_to_val:
                extent = int(shape_env.var_to_val[sym])
        if extent is None:
            var = getattr(info, "var", None)
            if var is not None and hasattr(var, "_sympy_"):
                sym = var._sympy_()
                if sym in shape_env.var_to_val:
                    extent = int(shape_env.var_to_val[sym])
        if extent is not None:
            loop_extents[info.block_id] = extent

    block_info_map = {info.block_id: info for info in bound_kernel.env.block_sizes}
    canonical: dict[tuple[frozenset[str], int], int] = {}
    alias: dict[int, int] = {}
    for grid_group in device_ir.grid_block_ids:
        for bid in grid_group:
            info = block_info_map.get(bid)
            if info is None:
                alias[bid] = bid
                continue
            key = (frozenset(info.debug_names), info.numel)
            canonical_id = canonical.setdefault(key, bid)
            alias[bid] = canonical_id
    for info in bound_kernel.env.block_sizes:
        alias.setdefault(info.block_id, info.block_id)

    used_block_ids: set[int] = set()
    for grp in device_ir.grid_block_ids:
        used_block_ids.update(grp)
    for ginfo in inner_graphs.values():
        used_block_ids.update(getattr(ginfo, "block_ids", []))
    used_canonical_ids = {alias.get(bid, bid) for bid in used_block_ids}

    module_attributes: dict[str, tuple[object, str]] = {}
    seen_canonical: set[int] = set()
    for info in bound_kernel.env.block_sizes:
        if isinstance(info.size, int):
            continue
        canonical_id = alias.get(info.block_id, info.block_id)
        if canonical_id not in used_canonical_ids or canonical_id in seen_canonical:
            continue
        seen_canonical.add(canonical_id)
        upper_bound = loop_extents.get(info.block_id)
        if upper_bound is None:
            continue
        sym_name = next(iter(info.debug_names), f"block_{canonical_id}")
        module_attributes[f"loom.{sym_name}"] = (
            f"{{upper_bound = {upper_bound} : index, is_reduction = {str(info.reduction).lower()}}}",
            "",
        )

    host_tensor_types: dict[str, str] = {}
    seen_host_tensors: set[str] = set()
    for graph_info in device_ir.graphs:
        if graph_info.graph_id in rolled_ids:
            continue
        for node in graph_info.graph.nodes:
            if node.op != "call_function" or node.target is not hl_tracing_ops._host_tensor:
                continue
            name = node.args[0]
            if name in seen_host_tensors:
                continue
            seen_host_tensors.add(name)
            fake_tensor = node.meta.get("val")
            if fake_tensor is None or not hasattr(fake_tensor, "shape"):
                continue
            dtype_str = torch_dtype_to_mlir_element_type(fake_tensor.dtype)
            shape: list[int | None] = []
            for dim_size in fake_tensor.shape:
                if hasattr(dim_size, "_sympy_"):
                    sym = dim_size._sympy_()
                    origin_info = bound_kernel.host_function.expr_to_origin.get(sym)
                    origin = origin_info.origin if origin_info else None
                    if isinstance(origin, BlockSizeOrigin):
                        block_info = bound_kernel.env.block_sizes[origin.block_id]
                        shape.append(block_info.size if isinstance(block_info.size, int) else None)
                    elif sym.is_number:
                        shape.append(int(sym))
                    elif sym in shape_env.var_to_val:
                        shape.append(int(shape_env.var_to_val[sym]))
                    else:
                        shape.append(None)
                elif isinstance(dim_size, int):
                    shape.append(int(dim_size))
                else:
                    shape.append(None)
            host_tensor_types[name] = format_memref_type(shape, dtype_str)

    stored_tensor_names = _collect_stored_tensor_names(
        device_ir.graphs,
        rolled_ids,
        host_tensor_target=hl_tracing_ops._host_tensor,
        store_target=hl_memory_ops.store,
    )
    ordered_tensor_names = _order_host_tensor_names(
        host_tensor_types,
        arg_types,
        stored_tensor_names,
    )

    all_parallel_block_ids: set[int] = set()
    for grp in device_ir.grid_block_ids:
        all_parallel_block_ids.update(grp)
    reduction_block_ids: list[int] = []
    for ginfo in inner_graphs.values():
        for bid in getattr(ginfo, "block_ids", []):
            if bid in all_parallel_block_ids or bid in reduction_block_ids:
                continue
            reduction_block_ids.append(bid)

    return KernelAnalysis(
        bound_kernel=bound_kernel,
        kernel_name=bound_kernel.kernel.fn.__name__,
        graph_inventory=GraphInventory(
            root_ids=root_ids,
            root_graphs=root_graphs,
            inner_graphs=inner_graphs,
            reachable_inner_ids=frozenset(reachable_inner_ids),
        ),
        block_info=BlockInfoSummary(
            canonical_aliases=alias,
            loop_extents=loop_extents,
            used_block_ids=frozenset(used_block_ids),
            used_canonical_block_ids=frozenset(used_canonical_ids),
        ),
        host_tensors=HostTensorInfo(
            tensor_types=host_tensor_types,
            arg_types=arg_types,
            ordered_tensor_names=ordered_tensor_names,
            stored_tensor_names=stored_tensor_names,
        ),
        module_attributes=module_attributes,
        reduction_block_ids=tuple(reduction_block_ids),
        assume_divisible_tiles=assume_divisible_tiles,
    )
