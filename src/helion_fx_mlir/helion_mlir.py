"""MLIR emission helpers for the staged Helion lowering prototype.

The goal is to make the textual IR evolve alongside project.plan.md.  The
stage-0 milestone now threads real metadata from a bound Helion kernel into the
generated MLIR so contributors can inspect loop-carried values, loads, stores,
and placeholder torch calls before the full DeviceIR pipeline is wired in.
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Iterable, Sequence, TYPE_CHECKING

import torch
import helion.language.memory_ops as hl_memory_ops
import helion.language._tracing_ops as hl_tracing_ops
from torch.ops import aten

if TYPE_CHECKING:
    from helion._compiler.device_ir import DeviceIR
    from helion._compiler.device_ir import GraphInfo
    from helion.runtime.kernel import BoundKernel
    from torch import Tensor

MLIR_OPT_FALLBACK = Path("/mnt/fast/llvm-mlir/bin/mlir-opt")


class _MLIRBuilder:
    """Tiny helper to manage indentation and SSA name creation."""

    def __init__(self) -> None:
        self._lines: list[str] = []
        self._indent = 0
        self._tmp_counter = 0

    def emit(self, text: str) -> None:
        self._lines.append("  " * self._indent + text)

    def push(self) -> None:
        self._indent += 1

    def pop(self) -> None:
        assert self._indent > 0
        self._indent -= 1

    def fresh(self, hint: str = "tmp") -> str:
        name = f"%{hint}{self._tmp_counter}"
        self._tmp_counter += 1
        return name

    def build(self) -> str:
        return "\n".join(self._lines) + "\n"


def generate_plan_stage0_mlir(
    bound_kernel: "BoundKernel",
    *,
    kernel_name: str = "helion_matmul_plan_stage0",
) -> str:
    """Generate the stage-0 MLIR skeleton using real Helion metadata."""

    fake_args = bound_kernel.fake_args
    if len(fake_args) < 2:
        raise ValueError("Expected the kernel to expose at least two tensor arguments.")

    lhs, rhs, *_ = fake_args

    block_sizes = {info.block_id: info for info in bound_kernel.env.block_sizes}
    grid_block_groups: Sequence[Sequence[int]] = bound_kernel.host_function.device_ir.grid_block_ids
    if not grid_block_groups:
        raise ValueError("device_ir.grid_block_ids is empty; nothing to lower.")
    parallel_block_ids = list(grid_block_groups[0])

    outer_loops: list[dict[str, object]] = []
    for block_id in parallel_block_ids:
        info = block_sizes[block_id]
        block_name = _first_debug_name(info.debug_names, fallback=f"block_{block_id}")
        total_extent = _resolve_extent(block_name, lhs, rhs)
        tile_size = int(info.size)
        trip_count = max(1, math.ceil(total_extent / tile_size))
        outer_loops.append(
            {
                "block_id": block_id,
                "name": block_name,
                "tile_size": tile_size,
                "trip_count": trip_count,
                "total_extent": total_extent,
            }
        )

    if not outer_loops:
        raise ValueError("No outer tile loops discovered in grid_block_ids.")

    reduction_block_ids = [
        block_id
        for block_id in _collect_reduction_block_ids(bound_kernel.host_function.device_ir)
        if block_id not in parallel_block_ids
    ]
    reduction_loops: list[dict[str, object]] = []
    for block_id in reduction_block_ids:
        info = block_sizes[block_id]
        block_name = _first_debug_name(info.debug_names, fallback=f"block_{block_id}")
        total_extent = _resolve_extent(block_name, lhs, rhs)
        tile_size = int(info.size)
        trip_count = max(1, math.ceil(total_extent / tile_size))
        reduction_loops.append(
            {
                "block_id": block_id,
                "name": block_name,
                "tile_size": tile_size,
                "trip_count": trip_count,
                "total_extent": total_extent,
            }
        )

    element_type = _torch_dtype_to_mlir_element_type(lhs.dtype)
    func_tensor_type = _format_tensor_type([None, None], element_type)
    full_shape = [_as_optional_int(lhs.size(0)), _as_optional_int(rhs.size(1))]
    full_shape_attr = _format_shape_attr(full_shape)

    tile_shape = [_as_optional_int(outer_loops[0]["tile_size"])]
    if len(outer_loops) > 1:
        tile_shape.append(_as_optional_int(outer_loops[1]["tile_size"]))
    tile_shape_attr = _format_shape_attr(tile_shape)

    builder = _MLIRBuilder()
    builder.emit("module {")
    builder.push()
    builder.emit(
        f"func.func @{kernel_name}(%arg0: {func_tensor_type}, %arg1: {func_tensor_type}) -> {func_tensor_type} {{"
    )
    builder.push()

    out_value = builder.fresh("out")
    builder.emit(
        f'{out_value} = "helion.alloc_like"(%arg0) {{shape = {full_shape_attr}}} : ({func_tensor_type}) -> {func_tensor_type}'
    )

    acc_seed = builder.fresh("acc_init")
    builder.emit(
        f'{acc_seed} = "helion.zero_tile"() {{shape = {tile_shape_attr}, dtype = "{element_type}"}} : () -> {func_tensor_type}'
    )

    c0 = _emit_index_constant(builder, 0)
    c1 = _emit_index_constant(builder, 1)
    dim_m = builder.fresh("dim_m")
    builder.emit(f"{dim_m} = tensor.dim %arg0, {c0} : {func_tensor_type}")
    dim_k = builder.fresh("dim_k")
    builder.emit(f"{dim_k} = tensor.dim %arg0, {c1} : {func_tensor_type}")
    dim_n = builder.fresh("dim_n")
    builder.emit(f"{dim_n} = tensor.dim %arg1, {c1} : {func_tensor_type}")

    builder.emit(
        f'"helion.annotate_tensor"(%arg0, {dim_m}, {dim_k}) {{name = "x"}} : ({func_tensor_type}, index, index) -> ()'
    )
    builder.emit(
        f'"helion.annotate_tensor"(%arg1, {dim_k}, {dim_n}) {{name = "y"}} : ({func_tensor_type}, index, index) -> ()'
    )

    iv_names = [f"%{loop['name']}_iv" for loop in outer_loops]
    zero_bounds = ", ".join("0" for _ in outer_loops)
    upper_bounds = ", ".join(str(loop["trip_count"]) for loop in outer_loops)
    steps = ", ".join("1" for _ in outer_loops)

    has_tile_b = any(loop["name"] == "tile_b" for loop in outer_loops)
    dims_map = {
        "tile_n": dim_n,
        "n": dim_n,
        "tile_k": dim_k,
        "k": dim_k,
    }
    if has_tile_b:
        dims_map["tile_b"] = dim_m
        dims_map["b"] = dim_m
        dims_map["tile_m"] = dim_k
    else:
        dims_map["tile_m"] = dim_m
        dims_map["m"] = dim_m

    for loop in outer_loops:
        loop_dim = dims_map.get(loop["name"])
        if loop_dim is not None:
            tile_const = builder.fresh(f"{loop['name']}_tile")
            builder.emit(f"{tile_const} = arith.constant {loop['tile_size']} : index")
            loop["tile_const"] = tile_const
            trip_count_ssa = builder.fresh(f"{loop['name']}_tiles")
            builder.emit(
                f"{trip_count_ssa} = arith.ceildivsi {loop_dim}, {tile_const} : index"
            )
            loop["trip_count_ssa"] = trip_count_ssa
        else:
            loop["trip_count_ssa"] = str(loop["trip_count"])
        builder.emit(
            f"// block_id={loop['block_id']} {loop['name']} size={loop['tile_size']} extent={loop['total_extent']} tiles={loop['trip_count']}"
        )
    builder.emit(
        f"affine.parallel ({', '.join(iv_names)}) = ({zero_bounds}) to ({', '.join(str(loop.get('trip_count_ssa', loop['trip_count'])) for loop in outer_loops)}) step ({steps}) {{"
    )
    builder.push()

    outer_tile_acc = acc_seed
    current_acc = outer_tile_acc
    outer_iv_m = iv_names[0] if iv_names else "%tile_m_iv"
    outer_iv_n = iv_names[1] if len(iv_names) > 1 else "%tile_n_iv"
    loop_map = {loop["name"]: loop for loop in outer_loops}

    outer_tile_sizes: dict[str, str] = {}
    for idx, loop in enumerate(outer_loops):
        loop_dim = dims_map.get(loop["name"])
        tile_const = loop.get("tile_const")
        if loop_dim is not None and tile_const is not None:
            iv_name = iv_names[idx]
            actual = _emit_dynamic_tile_size(
                builder, iv_name, loop_dim, tile_const, loop["name"]
            )
            outer_tile_sizes[loop["name"]] = actual
        elif tile_const is not None:
            outer_tile_sizes[loop["name"]] = tile_const

    for_graph = _first_graph_with_block_ids(bound_kernel.host_function.device_ir)
    fx_names = _extract_fx_names(for_graph) if for_graph is not None else {}
    root_graph = _first_root_graph(bound_kernel.host_function.device_ir)
    root_fx_info = _extract_root_fx_info(root_graph)

    for loop in reduction_loops:
        builder.emit(
            f"// block_id={loop['block_id']} {loop['name']} size={loop['tile_size']} extent={loop['total_extent']} tiles={loop['trip_count']}"
        )
        reduction_iv = f"%{loop['name']}_iv"
        loop_result = builder.fresh(f"{loop['name']}_acc")
        loop_map[loop["name"]] = loop
        loop_dim = dims_map.get(loop["name"])
        trip_bound = str(loop["trip_count"])
        if loop_dim is not None:
            tile_const = builder.fresh(f"{loop['name']}_tile")
            builder.emit(f"{tile_const} = arith.constant {loop['tile_size']} : index")
            loop["tile_const"] = tile_const
            trip_count_ssa = builder.fresh(f"{loop['name']}_tiles")
            builder.emit(
                f"{trip_count_ssa} = arith.ceildivsi {loop_dim}, {tile_const} : index"
            )
            loop["trip_count_ssa"] = trip_count_ssa
            trip_bound = trip_count_ssa
        else:
            loop["trip_count_ssa"] = str(loop["trip_count"])
        builder.emit(
            f"{loop_result} = affine.for {reduction_iv} = 0 to {trip_bound} "
            f"iter_args(%acc_iter = {current_acc}) -> ({func_tensor_type}) {{"
        )
        builder.push()

        tile_k_size = loop.get("tile_const")
        if loop_dim is not None and loop.get("tile_const") is not None:
            tile_k_size = _emit_dynamic_tile_size(
                builder, reduction_iv, loop_dim, loop["tile_const"], loop["name"]
            )

        lhs_tile_m_size = _choose_tile_size(builder, outer_tile_sizes, loop_map, "tile_m")
        lhs_tile_k_size = tile_k_size or loop.get("tile_const") or str(loop["tile_size"])
        lhs_tile = builder.fresh("lhs")
        lhs_indices = _format_indices_attr(
            [outer_iv_m, reduction_iv],
        )
        lhs_meta = _format_dynamic_tensor_meta(lhs_tile_m_size, lhs_tile_k_size, element_type)
        lhs_fx_attr = fx_names.get("lhs_load")
        builder.emit(
            f'{lhs_tile} = "helion.load_tile_dynamic"(%arg0, {lhs_tile_m_size}, {lhs_tile_k_size}) {{tile = {lhs_indices}, sizes = {tile_shape_attr}, '
            f"tensor_meta = {lhs_meta}"
            + (
                f", fx_node = {_format_string_attr(lhs_fx_attr)}"
                if lhs_fx_attr is not None
                else ""
            )
            + f"}} : ({func_tensor_type}, index, index) -> {func_tensor_type}"
        )

        rhs_tile_n_size = _choose_tile_size(builder, outer_tile_sizes, loop_map, "tile_n")
        rhs_tile = builder.fresh("rhs")
        rhs_indices = _format_indices_attr(
            [reduction_iv, outer_iv_n],
        )
        rhs_meta = _format_dynamic_tensor_meta(lhs_tile_k_size, rhs_tile_n_size, element_type)
        rhs_fx_attr = fx_names.get("rhs_load")
        builder.emit(
            f'{rhs_tile} = "helion.load_tile_dynamic"(%arg1, {lhs_tile_k_size}, {rhs_tile_n_size}) {{tile = {rhs_indices}, sizes = {tile_shape_attr}, '
            f"tensor_meta = {rhs_meta}"
            + (
                f", fx_node = {_format_string_attr(rhs_fx_attr)}"
                if rhs_fx_attr is not None
                else ""
            )
            + f"}} : ({func_tensor_type}, index, index) -> {func_tensor_type}"
        )

        acc_next = builder.fresh("acc")
        addmm_fx_attr = fx_names.get("addmm")
        builder.emit(
            f'{acc_next} = "helion.call_torch"(%acc_iter, {lhs_tile}, {rhs_tile}) '
            f'{{fn_name = "aten.addmm"'
            + (
                f", fx_node = {_format_string_attr(addmm_fx_attr)}"
                if addmm_fx_attr is not None
                else ""
            )
            + f"}} : ({func_tensor_type}, {func_tensor_type}, {func_tensor_type}) -> {func_tensor_type}"
        )
        builder.emit(f"affine.yield {acc_next} : {func_tensor_type}")
        builder.pop()
        builder.emit("}")
        current_acc = loop_result

    phi_fx_name = root_fx_info.get("phi")
    if phi_fx_name is not None:
        phi_result = builder.fresh("phi")
        builder.emit(
            f'{phi_result} = "helion.phi"({acc_seed}, {current_acc}) '
            + f'{{fx_node = {_format_string_attr(phi_fx_name)}}} '
            + f": ({func_tensor_type}, {func_tensor_type}) -> {func_tensor_type}"
        )
        current_acc = phi_result

    store_tile_m_size = _choose_tile_size(builder, outer_tile_sizes, loop_map, "tile_m")
    store_tile_n_size = _choose_tile_size(builder, outer_tile_sizes, loop_map, "tile_n")
    store_meta = _format_dynamic_tensor_meta(store_tile_m_size, store_tile_n_size, element_type)
    builder.emit(
        f'"helion.store_tile_dynamic"({out_value}, {current_acc}, {store_tile_m_size}, {store_tile_n_size}) {{tile = {_format_indices_attr([outer_iv_m, outer_iv_n])}, '
        f"sizes = {tile_shape_attr}, tensor_meta = {store_meta}"
        + (
            f", fx_node = {_format_string_attr(root_fx_info.get('store'))}"
            if root_fx_info.get("store") is not None
            else ""
        )
        + f"}} "
        f": ({func_tensor_type}, {func_tensor_type}, index, index) -> ()"
    )

    builder.emit("affine.yield")
    builder.pop()
    builder.emit("}")
    builder.emit(f"return {out_value} : {func_tensor_type}")
    builder.pop()
    builder.emit("}")
    builder.pop()
    builder.emit("}")

    return builder.build()


def validate_with_mlir_opt(
    mlir_text: str,
    *,
    mlir_opt_path: str | Path | None = None,
    extra_args: Iterable[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run `mlir-opt` to confirm the emitted IR parses."""

    if mlir_opt_path is None:
        mlir_opt_path = MLIR_OPT_FALLBACK

    tool = Path(mlir_opt_path)
    if not tool.exists():
        raise FileNotFoundError(
            f"mlir-opt not found at {tool}. Adjust MLIR_OPT_PATH or install MLIR."
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


def _collect_reduction_block_ids(device_ir: "DeviceIR") -> list[int]:
    block_ids: list[int] = []
    for graph_info in device_ir.graphs:
        candidate = getattr(graph_info, "block_ids", None)
        if candidate:
            for block_id in candidate:
                if block_id not in block_ids:
                    block_ids.append(block_id)
    return block_ids


def _first_debug_name(names: "set[str]", *, fallback: str) -> str:
    for name in names:
        if name:
            return name.replace(".", "_").replace("-", "_")
    return fallback


def _resolve_extent(name: str, lhs: "Tensor", rhs: "Tensor") -> int:
    if name in {"tile_m", "m"}:
        return int(lhs.size(0))
    if name in {"tile_b", "b"}:
        return int(lhs.size(0))
    if name in {"tile_n", "n"}:
        return int(rhs.size(1))
    if name in {"tile_k", "k"}:
        return int(lhs.size(1))
    raise ValueError(f"Cannot resolve extent for loop named '{name}'.")


def _torch_dtype_to_mlir_element_type(dtype: "torch.dtype") -> str:
    mapping = {
        torch.float16: "f16",
        torch.bfloat16: "bf16",
        torch.float32: "f32",
        torch.float64: "f64",
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype for MLIR emission: {dtype}")
    return mapping[dtype]


def _as_optional_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except Exception:
        return None


def _format_tensor_type(shape: Sequence[int | None], element_type: str) -> str:
    if not shape:
        return f"tensor<{element_type}>"
    dims = "x".join("?" if dim is None else str(dim) for dim in shape)
    return f"tensor<{dims}x{element_type}>"


def _format_shape_attr(shape: Sequence[int | None]) -> str:
    if not shape:
        return '"[]"'
    body = ", ".join("?" if dim is None else str(dim) for dim in shape)
    return f'"[{body}]"'


def _format_indices_attr(indices: Sequence[str | None]) -> str:
    if not indices:
        return '"[]"'
    body = ", ".join(index if index is not None else "?" for index in indices)
    return f'"[{body}]"'


def _format_tensor_meta(tensor: "Tensor", *, override_shape: Sequence[int | None] | None = None) -> str:
    if override_shape is None:
        override_shape = [_as_optional_int(dim) for dim in tensor.size()]
    shape_str = _format_shape_attr(list(override_shape))
    dtype_str = _torch_dtype_to_mlir_element_type(tensor.dtype)
    return f"{shape_str}:{dtype_str}"


def _format_string_attr(value: str | None) -> str:
    if value is None:
        return '""'
    escaped = value.replace('"', '\\"')
    return f'"{escaped}"'


def _emit_index_constant(builder: _MLIRBuilder, value: int) -> str:
    name = builder.fresh(f"c{value}_")
    builder.emit(f"{name} = arith.constant {value} : index")
    return name


def _format_dynamic_tensor_meta(
    dim0: str,
    dim1: str,
    element_type: str,
) -> str:
    return f'"[{dim0}, {dim1}]":{element_type}'


def _emit_dynamic_tile_size(
    builder: _MLIRBuilder,
    iv: str,
    dim: str,
    tile_const: str,
    hint: str,
) -> str:
    offset = builder.fresh(f"{hint}_offset")
    builder.emit(f"{offset} = arith.muli {iv}, {tile_const} : index")
    remain = builder.fresh(f"{hint}_remain")
    builder.emit(f"{remain} = arith.subi {dim}, {offset} : index")
    cmp = builder.fresh(f"{hint}_cmp")
    builder.emit(f"{cmp} = arith.cmpi slt, {remain}, {tile_const} : index")
    size = builder.fresh(f"{hint}_size")
    builder.emit(f"{size} = arith.select {cmp}, {remain}, {tile_const} : index")
    return size


def _choose_tile_size(
    builder: _MLIRBuilder,
    dynamic_sizes: dict[str, str],
    loop_map: dict[str, dict[str, object]],
    key: str,
) -> str:
    if key in dynamic_sizes:
        return dynamic_sizes[key]
    fallback_loop = loop_map.get(key)
    if fallback_loop:
        if "tile_const" in fallback_loop:
            return str(fallback_loop["tile_const"])
        const = _emit_index_constant(builder, int(fallback_loop.get("tile_size", 0)))
        fallback_loop["tile_const"] = const
        return const
    return _emit_index_constant(builder, 0)


def _first_graph_with_block_ids(device_ir: "DeviceIR") -> "GraphInfo | None":
    for graph_info in device_ir.graphs:
        if getattr(graph_info, "block_ids", None):
            return graph_info
    return None


def _first_root_graph(device_ir: "DeviceIR") -> "GraphInfo | None":
    for graph_info in device_ir.graphs:
        if getattr(graph_info, "block_ids", None):
            continue
        return graph_info
    return None


def _extract_fx_names(graph_info: "GraphInfo | None") -> dict[str, str]:
    if graph_info is None:
        return {}
    names: dict[str, str] = {}
    load_seen = 0
    for node in graph_info.graph.nodes:
        if node.op == "call_function" and node.target is hl_memory_ops.load:
            key = "lhs_load" if load_seen == 0 else "rhs_load"
            names[key] = node.name
            load_seen += 1
        elif node.op == "call_function" and node.target is aten.addmm.default:
            names["addmm"] = node.name
    return names


def _extract_root_fx_info(graph_info: "GraphInfo | None") -> dict[str, str]:
    info: dict[str, str] = {}
    if graph_info is None:
        return info
    for node in graph_info.graph.nodes:
        if node.op == "call_function":
            if node.target is hl_memory_ops.store:
                info["store"] = node.name
            elif node.target is hl_tracing_ops._phi:
                info["phi"] = node.name
    return info
