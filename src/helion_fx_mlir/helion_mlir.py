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
from typing import Iterable, Sequence, TYPE_CHECKING, Any

import torch
import helion.language.memory_ops as hl_memory_ops
import helion.language._tracing_ops as hl_tracing_ops
from torch.ops import aten

# Import BlockSizeInfo types for symbolic size detection
try:
    from helion._compiler.compile_environment import AutoSize
except ImportError:
    AutoSize = None  # type: ignore[misc,assignment]

if TYPE_CHECKING:
    from helion._compiler.device_ir import DeviceIR
    from helion._compiler.device_ir import GraphInfo
    from helion._compiler.compile_environment import BlockSizeInfo
    from helion.runtime.kernel import BoundKernel
    from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[2]
HELION_OPT_CANDIDATES = [
    REPO_ROOT / "build" / "mlir" / "helion-opt",
    REPO_ROOT / "build" / "bin" / "helion-opt",
    Path("/mnt/fast/llvm-mlir/bin/helion-opt"),
    Path("/mnt/fast/llvm-mlir/bin/mlir-opt"),
]


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


def _is_concrete_size(size: Any) -> bool:
    """Check if a block size is a concrete integer value.
    
    Returns True for int, False for SymInt, AutoSize, or None.
    """
    if size is None:
        return False
    if AutoSize is not None and isinstance(size, AutoSize):
        return False
    # Check for torch.SymInt - it has a special type
    if hasattr(torch, 'SymInt') and isinstance(size, torch.SymInt):
        return False
    # Also check for sympy expressions that may appear
    try:
        int(size)
        return True
    except (TypeError, ValueError):
        return False


def generate_plan_stage0_mlir(
    bound_kernel: "BoundKernel",
    *,
    kernel_name: str = "helion_matmul_plan_stage0",
) -> str:
    """Generate the stage-0 MLIR skeleton using real Helion metadata.
    
    Tile sizes are handled as follows:
    - Concrete int values: emitted as MLIR constants
    - Symbolic values (SymInt, AutoSize, None): emitted as function arguments
    """

    fake_args = bound_kernel.fake_args
    if len(fake_args) < 2:
        raise ValueError("Expected the kernel to expose at least two tensor arguments.")

    lhs, rhs, *_ = fake_args

    block_sizes = {info.block_id: info for info in bound_kernel.env.block_sizes}
    grid_block_groups: Sequence[Sequence[int]] = bound_kernel.host_function.device_ir.grid_block_ids
    if not grid_block_groups:
        raise ValueError("device_ir.grid_block_ids is empty; nothing to lower.")
    parallel_block_ids = list(grid_block_groups[0])

    # Collect symbolic tile size arguments
    symbolic_tile_args: list[dict[str, object]] = []
    
    outer_loops: list[dict[str, object]] = []
    for block_id in parallel_block_ids:
        info = block_sizes[block_id]
        block_name = _first_debug_name(info.debug_names, fallback=f"block_{block_id}")
        total_extent = _resolve_extent(block_name, lhs, rhs)
        
        if _is_concrete_size(info.size):
            tile_size = int(info.size)
            trip_count = max(1, math.ceil(total_extent / tile_size))
            is_symbolic = False
        else:
            # Symbolic tile size - will be a function argument
            tile_size = None  # Will be resolved at runtime
            trip_count = None  # Will be computed dynamically
            is_symbolic = True
            symbolic_tile_args.append({
                "block_id": block_id,
                "name": block_name,
                "arg_name": f"{block_name}_size",
            })
        
        outer_loops.append(
            {
                "block_id": block_id,
                "name": block_name,
                "tile_size": tile_size,
                "trip_count": trip_count,
                "total_extent": total_extent,
                "is_symbolic": is_symbolic,
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
        
        if _is_concrete_size(info.size):
            tile_size = int(info.size)
            trip_count = max(1, math.ceil(total_extent / tile_size))
            is_symbolic = False
        else:
            tile_size = None
            trip_count = None
            is_symbolic = True
            # Check if not already added from outer loops
            if not any(arg["name"] == block_name for arg in symbolic_tile_args):
                symbolic_tile_args.append({
                    "block_id": block_id,
                    "name": block_name,
                    "arg_name": f"{block_name}_size",
                })
        
        reduction_loops.append(
            {
                "block_id": block_id,
                "name": block_name,
                "tile_size": tile_size,
                "trip_count": trip_count,
                "total_extent": total_extent,
                "is_symbolic": is_symbolic,
            }
        )

    element_type = _torch_dtype_to_mlir_element_type(lhs.dtype)
    func_tensor_type = _format_tensor_type([None, None], element_type)
    full_shape = [_as_optional_int(lhs.size(0)), _as_optional_int(rhs.size(1))]
    full_shape_attr = _format_shape_attr(full_shape)

    # Tile shape may include symbolic values - use -1 for symbolic
    tile_shape = []
    for loop in outer_loops[:2]:
        if loop["is_symbolic"]:
            tile_shape.append(None)  # -1 in MLIR attr
        else:
            tile_shape.append(_as_optional_int(loop["tile_size"]))
    tile_shape_attr = _format_shape_attr(tile_shape)

    builder = _MLIRBuilder()
    builder.emit("module {")
    builder.push()
    
    # Build function signature with tensor args + symbolic tile size args
    func_args = [f"%arg0: {func_tensor_type}", f"%arg1: {func_tensor_type}"]
    symbolic_arg_ssa: dict[str, str] = {}  # name -> SSA value
    for idx, sym_arg in enumerate(symbolic_tile_args):
        arg_name = f"%{sym_arg['arg_name']}"
        func_args.append(f"{arg_name}: index")
        symbolic_arg_ssa[sym_arg["name"]] = arg_name
    
    func_args_str = ", ".join(func_args)
    builder.emit(
        f"func.func @{kernel_name}({func_args_str}) -> {func_tensor_type} {{"
    )
    builder.push()
    
    # Add comments for symbolic tile size arguments
    for sym_arg in symbolic_tile_args:
        builder.emit(f"// Symbolic tile size argument: {sym_arg['arg_name']} (block_id={sym_arg['block_id']})")

    out_value = builder.fresh("out")
    alloc_attrs = _format_attr_dict({"shape": full_shape_attr})
    builder.emit(
        f'{out_value} = "helion.alloc_like"(%arg0){alloc_attrs} : ({func_tensor_type}) -> {func_tensor_type}'
    )

    acc_seed = builder.fresh("acc_init")
    zero_attrs = _format_attr_dict({"shape": tile_shape_attr, "dtype": element_type})
    builder.emit(
        f'{acc_seed} = "helion.zero_tile"(){zero_attrs} : () -> {func_tensor_type}'
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
    
    # Check if we have any symbolic loops - if so, we need to use SCF dialect
    # instead of affine dialect since affine requires static bounds
    has_symbolic_loops = (
        any(loop.get("is_symbolic", False) for loop in outer_loops) or
        any(loop.get("is_symbolic", False) for loop in reduction_loops)
    )

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
        loop_name = loop["name"]
        is_symbolic = loop.get("is_symbolic", False)
        
        if is_symbolic:
            # Use the symbolic argument as the tile size
            tile_ssa = symbolic_arg_ssa.get(loop_name)
            if tile_ssa is None:
                raise ValueError(f"Missing symbolic argument for loop {loop_name}")
            loop["tile_const"] = tile_ssa
            if loop_dim is not None:
                # Use affine.apply with ceildiv so the result is a valid affine symbol
                trip_count_ssa = builder.fresh(f"{loop_name}_tiles")
                builder.emit(
                    f"{trip_count_ssa} = affine.apply affine_map<()[s0, s1] -> (s0 ceildiv s1)>()[{loop_dim}, {tile_ssa}]"
                )
                loop["trip_count_ssa"] = trip_count_ssa
            else:
                # Fallback - should not happen for well-formed inputs
                loop["trip_count_ssa"] = tile_ssa
            builder.emit(
                f"// block_id={loop['block_id']} {loop_name} size=SYMBOLIC extent={loop['total_extent']} tiles=dynamic"
            )
        elif loop_dim is not None:
            tile_const = builder.fresh(f"{loop_name}_tile")
            builder.emit(f"{tile_const} = arith.constant {loop['tile_size']} : index")
            loop["tile_const"] = tile_const
            # Use affine.apply with ceildiv so the result is a valid affine symbol
            trip_count_ssa = builder.fresh(f"{loop_name}_tiles")
            builder.emit(
                f"{trip_count_ssa} = affine.apply affine_map<()[s0, s1] -> (s0 ceildiv s1)>()[{loop_dim}, {tile_const}]"
            )
            loop["trip_count_ssa"] = trip_count_ssa
            builder.emit(
                f"// block_id={loop['block_id']} {loop_name} size={loop['tile_size']} extent={loop['total_extent']} tiles={loop['trip_count']}"
            )
        else:
            loop["trip_count_ssa"] = str(loop["trip_count"])
            builder.emit(
                f"// block_id={loop['block_id']} {loop_name} size={loop['tile_size']} extent={loop['total_extent']} tiles={loop['trip_count']}"
            )
    # Emit the outer parallel loop using affine.parallel
    # affine.parallel can accept symbol operands for bounds - dynamic values are passed as symbols
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
        is_symbolic = loop.get("is_symbolic", False)
        if loop_dim is not None and tile_const is not None:
            iv_name = iv_names[idx]
            if is_symbolic:
                # For symbolic tile sizes, use the symbolic SSA value directly
                actual = _emit_dynamic_tile_size_symbolic(
                    builder, iv_name, loop_dim, tile_const, loop["name"]
                )
            else:
                actual = _emit_dynamic_tile_size(
                    builder, iv_name, loop_dim, loop["tile_size"], loop["name"]
                )
            outer_tile_sizes[loop["name"]] = actual
        elif tile_const is not None:
            outer_tile_sizes[loop["name"]] = tile_const

    for_graph = _first_graph_with_block_ids(bound_kernel.host_function.device_ir)
    fx_names = _extract_fx_names(for_graph) if for_graph is not None else {}
    root_graph = _first_root_graph(bound_kernel.host_function.device_ir)
    root_fx_info = _extract_root_fx_info(root_graph)

    for loop in reduction_loops:
        loop_name = loop["name"]
        is_symbolic = loop.get("is_symbolic", False)
        if is_symbolic:
            builder.emit(
                f"// block_id={loop['block_id']} {loop_name} size=SYMBOLIC extent={loop['total_extent']} tiles=dynamic"
            )
        else:
            builder.emit(
                f"// block_id={loop['block_id']} {loop_name} size={loop['tile_size']} extent={loop['total_extent']} tiles={loop['trip_count']}"
            )
        reduction_iv = f"%{loop_name}_iv"
        loop_result = builder.fresh(f"{loop_name}_acc")
        loop_map[loop_name] = loop
        loop_dim = dims_map.get(loop_name)
        
        if is_symbolic:
            # Use the symbolic argument as the tile size
            tile_ssa = symbolic_arg_ssa.get(loop_name)
            if tile_ssa is None:
                raise ValueError(f"Missing symbolic argument for reduction loop {loop_name}")
            loop["tile_const"] = tile_ssa
            if loop_dim is not None:
                # Use affine.apply with ceildiv so the result is a valid affine symbol
                trip_count_ssa = builder.fresh(f"{loop_name}_tiles")
                builder.emit(
                    f"{trip_count_ssa} = affine.apply affine_map<()[s0, s1] -> (s0 ceildiv s1)>()[{loop_dim}, {tile_ssa}]"
                )
                loop["trip_count_ssa"] = trip_count_ssa
                trip_bound = trip_count_ssa
            else:
                loop["trip_count_ssa"] = tile_ssa
                trip_bound = tile_ssa
        elif loop_dim is not None:
            tile_const = builder.fresh(f"{loop_name}_tile")
            builder.emit(f"{tile_const} = arith.constant {loop['tile_size']} : index")
            loop["tile_const"] = tile_const
            # Use affine.apply with ceildiv so the result is a valid affine symbol
            trip_count_ssa = builder.fresh(f"{loop_name}_tiles")
            builder.emit(
                f"{trip_count_ssa} = affine.apply affine_map<()[s0, s1] -> (s0 ceildiv s1)>()[{loop_dim}, {tile_const}]"
            )
            loop["trip_count_ssa"] = trip_count_ssa
            trip_bound = trip_count_ssa
        else:
            loop["trip_count_ssa"] = str(loop["trip_count"])
            trip_bound = str(loop["trip_count"])
        
        # Emit reduction loop using affine.for
        # affine.for can accept symbol operands for bounds - dynamic values are passed as symbols
        builder.emit(
            f"{loop_result} = affine.for {reduction_iv} = 0 to {trip_bound} "
            f"iter_args(%acc_iter = {current_acc}) -> ({func_tensor_type}) {{"
        )
        builder.push()

        tile_k_size = loop.get("tile_const")
        if loop_dim is not None and loop.get("tile_const") is not None:
            if is_symbolic:
                tile_k_size = _emit_dynamic_tile_size_symbolic(
                    builder, reduction_iv, loop_dim, loop["tile_const"], loop_name
                )
            else:
                tile_k_size = _emit_dynamic_tile_size(
                    builder, reduction_iv, loop_dim, loop["tile_size"], loop_name
                )

        lhs_tile_m_size = _choose_tile_size(builder, outer_tile_sizes, loop_map, "tile_m")
        lhs_tile_k_size = tile_k_size or loop.get("tile_const") or str(loop["tile_size"])
        lhs_tile = builder.fresh("lhs")
        lhs_indices = _format_indices_attr(
            [outer_iv_m, reduction_iv],
        )
        lhs_meta = _format_dynamic_tensor_meta(lhs_tile_m_size, lhs_tile_k_size, element_type)
        lhs_fx_attr = fx_names.get("lhs_load")
        lhs_attrs = _format_attr_dict(
            {
                "tile": lhs_indices,
                "sizes": tile_shape_attr,
                "tensor_meta": lhs_meta,
                "fx_node": _format_string_attr(lhs_fx_attr) if lhs_fx_attr is not None else None,
            }
        )
        builder.emit(
            f'{lhs_tile} = "helion.load_tile_dynamic"(%arg0, {lhs_tile_m_size}, {lhs_tile_k_size}){lhs_attrs} : ({func_tensor_type}, index, index) -> {func_tensor_type}'
        )

        rhs_tile_n_size = _choose_tile_size(builder, outer_tile_sizes, loop_map, "tile_n")
        rhs_tile = builder.fresh("rhs")
        rhs_indices = _format_indices_attr(
            [reduction_iv, outer_iv_n],
        )
        rhs_meta = _format_dynamic_tensor_meta(lhs_tile_k_size, rhs_tile_n_size, element_type)
        rhs_fx_attr = fx_names.get("rhs_load")
        rhs_attrs = _format_attr_dict(
            {
                "tile": rhs_indices,
                "sizes": tile_shape_attr,
                "tensor_meta": rhs_meta,
                "fx_node": _format_string_attr(rhs_fx_attr) if rhs_fx_attr is not None else None,
            }
        )
        builder.emit(
            f'{rhs_tile} = "helion.load_tile_dynamic"(%arg1, {lhs_tile_k_size}, {rhs_tile_n_size}){rhs_attrs} : ({func_tensor_type}, index, index) -> {func_tensor_type}'
        )

        acc_next = builder.fresh("acc")
        addmm_fx_attr = fx_names.get("addmm")
        call_attrs = _format_attr_dict(
            {
                "fn_name": _format_string_attr("aten.addmm"),
                "fx_node": _format_string_attr(addmm_fx_attr) if addmm_fx_attr is not None else None,
            }
        )
        builder.emit(
            f'{acc_next} = "helion.call_torch"(%acc_iter, {lhs_tile}, {rhs_tile}){call_attrs} : ({func_tensor_type}, {func_tensor_type}, {func_tensor_type}) -> {func_tensor_type}'
        )
        builder.emit(f"affine.yield {acc_next} : {func_tensor_type}")
        builder.pop()
        builder.emit("}")
        current_acc = loop_result

    phi_fx_name = root_fx_info.get("phi")
    if phi_fx_name is not None:
        phi_result = builder.fresh("phi")
        phi_attrs = _format_attr_dict({"fx_node": _format_string_attr(phi_fx_name)})
        builder.emit(
            f'{phi_result} = "helion.phi"({acc_seed}, {current_acc}){phi_attrs} : ({func_tensor_type}, {func_tensor_type}) -> {func_tensor_type}'
        )
        current_acc = phi_result

    store_tile_m_size = _choose_tile_size(builder, outer_tile_sizes, loop_map, "tile_m")
    store_tile_n_size = _choose_tile_size(builder, outer_tile_sizes, loop_map, "tile_n")
    store_meta = _format_dynamic_tensor_meta(store_tile_m_size, store_tile_n_size, element_type)
    store_attrs = _format_attr_dict(
        {
            "tile": _format_indices_attr([outer_iv_m, outer_iv_n]),
            "sizes": tile_shape_attr,
            "tensor_meta": store_meta,
            "fx_node": _format_string_attr(root_fx_info.get("store"))
            if root_fx_info.get("store") is not None
            else None,
        }
    )
    builder.emit(
        f'"helion.store_tile_dynamic"({out_value}, {current_acc}, {store_tile_m_size}, {store_tile_n_size}){store_attrs} '
        f": ({func_tensor_type}, {func_tensor_type}, index, index) -> ()"
    )

    # Terminate the parallel loop with affine.yield
    builder.emit("affine.yield")
    builder.pop()
    builder.emit("}")
    builder.emit(f"return {out_value} : {func_tensor_type}")
    builder.pop()
    builder.emit("}")
    builder.pop()
    builder.emit("}")

    return builder.build()


def validate_with_helion_opt(
    mlir_text: str,
    *,
    opt_path: str | Path | None = None,
    extra_args: Iterable[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run `helion-opt` (falling back to `mlir-opt`) to confirm the emitted IR parses."""

    tool_candidates: Iterable[Path] = HELION_OPT_CANDIDATES if opt_path is None else [Path(opt_path)]

    tool: Path | None = None
    for candidate in tool_candidates:
        if candidate.exists():
            tool = candidate
            break

    if tool is None:
        raise FileNotFoundError(
            "Unable to locate `helion-opt` or `mlir-opt`. "
            "Pass `mlir_opt_path` explicitly once the project is built."
        )

    args = [str(tool)]
    if tool.name == "mlir-opt":
        args.append("-allow-unregistered-dialect")
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
        return "[]"
    body = ", ".join(str(dim) if dim is not None else "-1" for dim in shape)
    return f"[{body}]"


def _format_indices_attr(indices: Sequence[str | None]) -> str:
    if not indices:
        return "[]"
    body = ", ".join(_format_string_attr(index if index is not None else "?") for index in indices)
    return f"[{body}]"


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


def _format_attr_dict(attrs: dict[str, str | None]) -> str:
    items = [f"{key} = {value}" for key, value in attrs.items() if value is not None]
    if not items:
        return ""
    return " {" + ", ".join(items) + "}"


def _emit_index_constant(builder: _MLIRBuilder, value: int) -> str:
    name = builder.fresh(f"c{value}_")
    builder.emit(f"{name} = arith.constant {value} : index")
    return name


def _format_dynamic_tensor_meta(
    dim0: str,
    dim1: str,
    element_type: str,
) -> str:
    return _format_string_attr(f"[{dim0}, {dim1}]:{element_type}")


def _emit_dynamic_tile_size(
    builder: _MLIRBuilder,
    iv: str,
    dim: str,
    tile_size: int,
    hint: str,
) -> str:
    size = builder.fresh(f"{hint}_size")
    map_str = f"affine_map<(d0)[s0] -> ({tile_size}, s0 - d0 * {tile_size})>"
    builder.emit(
        f"{size} = affine.min {map_str}({iv})[{dim}]"
    )
    return size


def _emit_dynamic_tile_size_symbolic(
    builder: _MLIRBuilder,
    iv: str,
    dim: str,
    tile_size_ssa: str,
    hint: str,
) -> str:
    """Emit a dynamic tile size computation for symbolic (SSA) tile sizes.
    
    Uses affine.min with the tile size as an affine symbol (s1).
    The affine_map is: (d0)[s0, s1] -> (s1, s0 - d0 * s1)
    where d0 is the loop IV, s0 is the dimension, and s1 is the tile size.
    
    The computation is: min(tile_size, dim - iv * tile_size)
    """
    size = builder.fresh(f"{hint}_size")
    # s0 = dim, s1 = tile_size
    # min(s1, s0 - d0 * s1)
    map_str = "affine_map<(d0)[s0, s1] -> (s1, s0 - d0 * s1)>"
    builder.emit(
        f"{size} = affine.min {map_str}({iv})[{dim}, {tile_size_ssa}]"
    )
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
        # For symbolic tile sizes, tile_size may be None - use tile_const if available
        tile_size = fallback_loop.get("tile_size")
        if tile_size is not None:
            const = _emit_index_constant(builder, int(tile_size))
            fallback_loop["tile_const"] = const
            return const
        # If tile_size is None (symbolic), we should already have tile_const set
        # This is a fallback that shouldn't normally be hit
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
