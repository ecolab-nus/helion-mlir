"""MLIR emission helpers for the staged Helion lowering prototype.

The goal is to make the textual IR evolve alongside project.plan.md. The
stage-0 milestone now threads real metadata from a bound Helion kernel into the
generated MLIR so contributors can inspect loop-carried values, loads, stores,
and placeholder torch calls before the full DeviceIR pipeline is wired in.

This module uses a modular lowering architecture:
- MLIRBuilder: Handles MLIR text emission and SSA naming
- LoweringContext: Holds state during lowering
- LoweringRegistry: Maps FX targets to lowering implementations
- lowerings/: Individual lowering implementations per op category
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable, Sequence, TYPE_CHECKING

import torch
import helion.language.memory_ops as hl_memory_ops
import helion.language._tracing_ops as hl_tracing_ops
from torch.ops import aten

# Import the modular lowering infrastructure
from .mlir_builder import (
    MLIRBuilder,
    is_concrete_size,
    torch_dtype_to_mlir_element_type,
    format_tensor_type,
    format_shape_attr,
    format_indices_attr,
    format_string_attr,
    format_attr_dict,
    format_dynamic_tensor_meta,
    as_optional_int,
)
from .lowering_context import (
    LoweringContext,
    LoopInfo,
    KernelArgInfo,
    first_debug_name,
    resolve_extent,
    collect_reduction_block_ids,
)
from .op_registry import LoweringRegistry

# Import lowerings to trigger registration
from . import lowerings  # noqa: F401

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


def generate_plan_stage0_mlir(
    bound_kernel: "BoundKernel",
    *,
    kernel_name: str = "helion_matmul_plan_stage0",
) -> str:
    """Generate the stage-0 MLIR skeleton using real Helion metadata.
    
    This function uses the modular lowering architecture to convert a bound
    Helion kernel to MLIR text. It handles:
    - Module and function structure
    - Parallel and reduction loop emission
    - FX node lowering via the registry
    
    Tile sizes are handled as follows:
    - Concrete int values: emitted as MLIR constants
    - Symbolic values (SymInt, AutoSize, None): emitted as function arguments
    """
    # Create lowering context from bound kernel
    ctx = LoweringContext.from_bound_kernel(bound_kernel, kernel_name)
    builder = ctx.builder
    
    # Get tensor arguments
    fake_args = bound_kernel.fake_args
    lhs, rhs, *_ = fake_args
    
    # Compute full output shape
    full_shape = [as_optional_int(lhs.size(0)), as_optional_int(rhs.size(1))]
    full_shape_attr = format_shape_attr(full_shape)
    
    # Emit module start with tile size attributes
    module_attrs = ctx.get_module_attributes()
    builder.emit_module_start(module_attrs)
    
    # Build function signature from kernel arguments only (no tile size args)
    func_args = ctx.get_func_signature_args()
    symbolic_tile_args = ctx.get_symbolic_tile_args()
    
    # Determine result type (use first tensor arg's type as template)
    tensor_args = ctx.get_tensor_args()
    result_type = tensor_args[0].mlir_type if tensor_args else ctx.tensor_type
    
    # Emit function start
    builder.emit_func_start(kernel_name, func_args, result_type)
    
    # Emit get_module_attribute ops for each symbolic tile size
    for sym_arg in symbolic_tile_args:
        loop_name = sym_arg["name"]
        attr_name = f"loom.{loop_name}"
        ssa = builder.emit_get_module_attribute(attr_name, f"{loop_name}_size")
        ctx.symbolic_arg_ssa[loop_name] = ssa
        builder.emit_comment(f"Tile size from module attribute: {attr_name} (block_id={sym_arg['block_id']})")
    
    # Emit output allocation (using first tensor arg as template)
    lhs_ssa = ctx.get_lhs_tensor_ssa()
    lhs_type = ctx.get_lhs_tensor_type()
    
    ctx.out_value = builder.fresh("out")
    alloc_attrs = format_attr_dict({"shape": full_shape_attr})
    builder.emit(
        f'{ctx.out_value} = "helion.alloc_like"({lhs_ssa}){alloc_attrs} : ({lhs_type}) -> {lhs_type}'
    )
    
    # Emit accumulator initialization
    ctx.acc_seed = builder.fresh("acc_init")
    zero_attrs = format_attr_dict({"shape": ctx.tile_shape_attr, "dtype": ctx.element_type})
    builder.emit(
        f'{ctx.acc_seed} = "helion.zero_tile"(){zero_attrs} : () -> {ctx.tensor_type}'
    )
    
    # Emit dimension queries
    _emit_dimension_queries(ctx)
    
    # Emit tensor annotations
    _emit_tensor_annotations(ctx)
    
    # Set up dimension mapping
    ctx.setup_dims_map()
    
    # Emit loop bounds computation and metadata
    _emit_loop_bounds(ctx)
    
    # Emit outer parallel loop
    _emit_parallel_loop_structure(ctx, bound_kernel)
    
    # Emit function end and return
    builder.emit(f"return {ctx.out_value} : {ctx.tensor_type}")
    builder.emit_func_end()
    builder.emit_module_end()
    
    return builder.build()


def _emit_dimension_queries(ctx: LoweringContext) -> None:
    """Emit tensor.dim operations for tensor dimensions."""
    builder = ctx.builder
    
    # Get tensor argument SSA names
    lhs_ssa = ctx.get_lhs_tensor_ssa()
    rhs_ssa = ctx.get_rhs_tensor_ssa()
    lhs_type = ctx.get_lhs_tensor_type()
    rhs_type = ctx.get_rhs_tensor_type()
    
    c0 = builder.emit_index_constant(0)
    c1 = builder.emit_index_constant(1)
    
    ctx.dim_m = builder.fresh("dim_m")
    builder.emit(f"{ctx.dim_m} = tensor.dim {lhs_ssa}, {c0} : {lhs_type}")
    
    ctx.dim_k = builder.fresh("dim_k")
    builder.emit(f"{ctx.dim_k} = tensor.dim {lhs_ssa}, {c1} : {lhs_type}")
    
    ctx.dim_n = builder.fresh("dim_n")
    builder.emit(f"{ctx.dim_n} = tensor.dim {rhs_ssa}, {c1} : {rhs_type}")


def _emit_tensor_annotations(ctx: LoweringContext) -> None:
    """Emit helion.annotate_tensor operations for input tensors."""
    builder = ctx.builder
    
    # Get tensor argument info
    tensor_args = ctx.get_tensor_args()
    
    if len(tensor_args) >= 1:
        lhs = tensor_args[0]
        builder.emit(
            f'"helion.annotate_tensor"({lhs.ssa_name}, {ctx.dim_m}, {ctx.dim_k}) {{name = "{lhs.name}"}} : ({lhs.mlir_type}, index, index) -> ()'
        )
    
    if len(tensor_args) >= 2:
        rhs = tensor_args[1]
        builder.emit(
            f'"helion.annotate_tensor"({rhs.ssa_name}, {ctx.dim_k}, {ctx.dim_n}) {{name = "{rhs.name}"}} : ({rhs.mlir_type}, index, index) -> ()'
        )


def _emit_loop_bounds(ctx: LoweringContext) -> None:
    """Emit loop bounds computation for outer and reduction loops."""
    builder = ctx.builder
    
    # Process outer loops
    for loop in ctx.outer_loops:
        loop_dim = ctx.dims_map.get(loop.name)
        
        if loop.is_symbolic:
            # Use the symbolic argument as the tile size
            tile_ssa = ctx.symbolic_arg_ssa.get(loop.name)
            if tile_ssa is None:
                raise ValueError(f"Missing symbolic argument for loop {loop.name}")
            loop.tile_const = tile_ssa
            
            if loop_dim is not None:
                # Compute trip count with ceildiv
                trip_count_ssa = builder.emit_affine_apply(
                    "()[s0, s1] -> (s0 ceildiv s1)",
                    [],
                    [loop_dim, tile_ssa],
                )
                loop.trip_count_ssa = trip_count_ssa
            else:
                loop.trip_count_ssa = tile_ssa
            
            builder.emit_comment(
                f"block_id={loop.block_id} {loop.name} size=SYMBOLIC extent={loop.total_extent} tiles=dynamic"
            )
        elif loop_dim is not None:
            # Emit tile size constant
            tile_const = builder.fresh(f"{loop.name}_tile")
            builder.emit(f"{tile_const} = arith.constant {loop.tile_size} : index")
            loop.tile_const = tile_const
            
            # Compute trip count with ceildiv
            trip_count_ssa = builder.emit_affine_apply(
                "()[s0, s1] -> (s0 ceildiv s1)",
                [],
                [loop_dim, tile_const],
            )
            loop.trip_count_ssa = trip_count_ssa
            
            builder.emit_comment(
                f"block_id={loop.block_id} {loop.name} size={loop.tile_size} extent={loop.total_extent} tiles={loop.trip_count}"
            )
        else:
            loop.trip_count_ssa = str(loop.trip_count)
            builder.emit_comment(
                f"block_id={loop.block_id} {loop.name} size={loop.tile_size} extent={loop.total_extent} tiles={loop.trip_count}"
            )


def _emit_parallel_loop_structure(ctx: LoweringContext, bound_kernel: "BoundKernel") -> None:
    """Emit the outer parallel loop and its contents."""
    builder = ctx.builder
    
    # Build IV names and bounds
    iv_names = [f"%{loop.name}_iv" for loop in ctx.outer_loops]
    for i, loop in enumerate(ctx.outer_loops):
        loop.iv_name = iv_names[i]
    
    zero_bounds = ["0"] * len(ctx.outer_loops)
    upper_bounds = [loop.trip_count_ssa or str(loop.trip_count) for loop in ctx.outer_loops]
    steps = ["1"] * len(ctx.outer_loops)
    
    # Emit parallel loop start
    builder.emit_affine_parallel_start(iv_names, zero_bounds, upper_bounds, steps)
    
    # Compute dynamic tile sizes for outer loops
    _emit_outer_tile_sizes(ctx)
    
    # Extract FX names from the device IR graphs
    _extract_fx_metadata(ctx, bound_kernel)
    
    # Emit reduction loops and body
    ctx.current_acc = ctx.acc_seed
    _emit_reduction_loops(ctx)
    
    # Emit phi node if present
    _emit_phi_if_present(ctx)
    
    # Emit store operation
    _emit_store_tile(ctx)
    
    # Emit parallel loop end
    builder.emit_affine_parallel_end()


def _emit_outer_tile_sizes(ctx: LoweringContext) -> None:
    """Compute and emit dynamic tile sizes for outer loops."""
    builder = ctx.builder
    
    for loop in ctx.outer_loops:
        loop_dim = ctx.dims_map.get(loop.name)
        
        if loop_dim is not None and loop.tile_const is not None:
            if loop.is_symbolic:
                actual = _emit_dynamic_tile_size_symbolic(
                    builder, loop.iv_name, loop_dim, loop.tile_const, loop.name
                )
            else:
                actual = _emit_dynamic_tile_size(
                    builder, loop.iv_name, loop_dim, loop.tile_size, loop.name
                )
            ctx.outer_tile_sizes[loop.name] = actual
        elif loop.tile_const is not None:
            ctx.outer_tile_sizes[loop.name] = loop.tile_const


def _extract_fx_metadata(ctx: LoweringContext, bound_kernel: "BoundKernel") -> None:
    """Extract FX node metadata from device IR graphs."""
    device_ir = bound_kernel.host_function.device_ir
    
    for_graph = _first_graph_with_block_ids(device_ir)
    ctx.fx_names = _extract_fx_names(for_graph) if for_graph is not None else {}
    
    root_graph = _first_root_graph(device_ir)
    ctx.root_fx_info = _extract_root_fx_info(root_graph)


def _emit_reduction_loops(ctx: LoweringContext) -> None:
    """Emit reduction loop(s) with loads and computation."""
    builder = ctx.builder
    loop_map = ctx.get_loop_map()
    
    outer_ivs = [loop.iv_name for loop in ctx.outer_loops]
    outer_iv_m = outer_ivs[0] if outer_ivs else "%tile_m_iv"
    outer_iv_n = outer_ivs[1] if len(outer_ivs) > 1 else "%tile_n_iv"
    
    for loop in ctx.reduction_loops:
        # Emit loop metadata comment
        if loop.is_symbolic:
            builder.emit_comment(
                f"block_id={loop.block_id} {loop.name} size=SYMBOLIC extent={loop.total_extent} tiles=dynamic"
            )
        else:
            builder.emit_comment(
                f"block_id={loop.block_id} {loop.name} size={loop.tile_size} extent={loop.total_extent} tiles={loop.trip_count}"
            )
        
        # Set up loop IV
        reduction_iv = f"%{loop.name}_iv"
        loop.iv_name = reduction_iv
        loop_result = builder.fresh(f"{loop.name}_acc")
        
        # Compute trip count
        loop_dim = ctx.dims_map.get(loop.name)
        trip_bound = _compute_reduction_trip_bound(ctx, loop, loop_dim)
        
        # Emit affine.for with iter_args
        builder.emit(
            f"{loop_result} = affine.for {reduction_iv} = 0 to {trip_bound} "
            f"iter_args(%acc_iter = {ctx.current_acc}) -> ({ctx.tensor_type}) {{"
        )
        builder.push()
        
        # Compute tile sizes for this iteration
        tile_k_size = _compute_reduction_tile_size(ctx, loop, loop_dim)
        
        # Emit LHS load
        lhs_tile = _emit_lhs_load(ctx, outer_iv_m, reduction_iv, tile_k_size)
        
        # Emit RHS load
        rhs_tile = _emit_rhs_load(ctx, reduction_iv, outer_iv_n, tile_k_size)
        
        # Emit computation (addmm)
        acc_next = _emit_addmm(ctx, lhs_tile, rhs_tile)
        
        # Emit yield
        builder.emit(f"affine.yield {acc_next} : {ctx.tensor_type}")
        builder.pop()
        builder.emit("}")
        
        ctx.current_acc = loop_result


def _compute_reduction_trip_bound(
    ctx: LoweringContext, loop: LoopInfo, loop_dim: str | None
) -> str:
    """Compute the trip bound for a reduction loop."""
    builder = ctx.builder
    
    if loop.is_symbolic:
        tile_ssa = ctx.symbolic_arg_ssa.get(loop.name)
        if tile_ssa is None:
            raise ValueError(f"Missing symbolic argument for reduction loop {loop.name}")
        loop.tile_const = tile_ssa
        
        if loop_dim is not None:
            trip_count_ssa = builder.emit_affine_apply(
                "()[s0, s1] -> (s0 ceildiv s1)",
                [],
                [loop_dim, tile_ssa],
            )
            loop.trip_count_ssa = trip_count_ssa
            return trip_count_ssa
        else:
            loop.trip_count_ssa = tile_ssa
            return tile_ssa
    elif loop_dim is not None:
        tile_const = builder.fresh(f"{loop.name}_tile")
        builder.emit(f"{tile_const} = arith.constant {loop.tile_size} : index")
        loop.tile_const = tile_const
        
        trip_count_ssa = builder.emit_affine_apply(
            "()[s0, s1] -> (s0 ceildiv s1)",
            [],
            [loop_dim, tile_const],
        )
        loop.trip_count_ssa = trip_count_ssa
        return trip_count_ssa
    else:
        loop.trip_count_ssa = str(loop.trip_count)
        return str(loop.trip_count)


def _compute_reduction_tile_size(
    ctx: LoweringContext, loop: LoopInfo, loop_dim: str | None
) -> str:
    """Compute the tile size for the current reduction iteration."""
    builder = ctx.builder
    
    if loop_dim is not None and loop.tile_const is not None:
        if loop.is_symbolic:
            return _emit_dynamic_tile_size_symbolic(
                builder, loop.iv_name, loop_dim, loop.tile_const, loop.name
            )
        else:
            return _emit_dynamic_tile_size(
                builder, loop.iv_name, loop_dim, loop.tile_size, loop.name
            )
    return loop.tile_const or str(loop.tile_size)


def _emit_lhs_load(
    ctx: LoweringContext, outer_iv_m: str, reduction_iv: str, tile_k_size: str
) -> str:
    """Emit LHS tile load operation."""
    builder = ctx.builder
    loop_map = ctx.get_loop_map()
    
    # Get LHS tensor info
    lhs_ssa = ctx.get_lhs_tensor_ssa()
    lhs_type = ctx.get_lhs_tensor_type()
    
    lhs_tile_m_size = _choose_tile_size(builder, ctx.outer_tile_sizes, loop_map, "tile_m")
    lhs_tile = builder.fresh("lhs")
    
    lhs_indices = format_indices_attr([outer_iv_m, reduction_iv])
    lhs_meta = format_dynamic_tensor_meta(lhs_tile_m_size, tile_k_size, ctx.element_type)
    lhs_fx_attr = ctx.fx_names.get("lhs_load")
    
    lhs_attrs = format_attr_dict({
        "tile": lhs_indices,
        "sizes": ctx.tile_shape_attr,
        "tensor_meta": lhs_meta,
        "fx_node": format_string_attr(lhs_fx_attr) if lhs_fx_attr else None,
    })
    
    builder.emit(
        f'{lhs_tile} = "helion.load_tile_dynamic"({lhs_ssa}, {lhs_tile_m_size}, {tile_k_size}){lhs_attrs} '
        f": ({lhs_type}, index, index) -> {lhs_type}"
    )
    
    return lhs_tile


def _emit_rhs_load(
    ctx: LoweringContext, reduction_iv: str, outer_iv_n: str, tile_k_size: str
) -> str:
    """Emit RHS tile load operation."""
    builder = ctx.builder
    loop_map = ctx.get_loop_map()
    
    # Get RHS tensor info
    rhs_ssa = ctx.get_rhs_tensor_ssa()
    rhs_type = ctx.get_rhs_tensor_type()
    
    rhs_tile_n_size = _choose_tile_size(builder, ctx.outer_tile_sizes, loop_map, "tile_n")
    rhs_tile = builder.fresh("rhs")
    
    rhs_indices = format_indices_attr([reduction_iv, outer_iv_n])
    rhs_meta = format_dynamic_tensor_meta(tile_k_size, rhs_tile_n_size, ctx.element_type)
    rhs_fx_attr = ctx.fx_names.get("rhs_load")
    
    rhs_attrs = format_attr_dict({
        "tile": rhs_indices,
        "sizes": ctx.tile_shape_attr,
        "tensor_meta": rhs_meta,
        "fx_node": format_string_attr(rhs_fx_attr) if rhs_fx_attr else None,
    })
    
    builder.emit(
        f'{rhs_tile} = "helion.load_tile_dynamic"({rhs_ssa}, {tile_k_size}, {rhs_tile_n_size}){rhs_attrs} '
        f": ({rhs_type}, index, index) -> {rhs_type}"
    )
    
    return rhs_tile


def _emit_addmm(ctx: LoweringContext, lhs_tile: str, rhs_tile: str) -> str:
    """Emit aten.addmm via helion.call_torch."""
    builder = ctx.builder
    
    acc_next = builder.fresh("acc")
    addmm_fx_attr = ctx.fx_names.get("addmm")
    
    call_attrs = format_attr_dict({
        "fn_name": format_string_attr("aten.addmm"),
        "fx_node": format_string_attr(addmm_fx_attr) if addmm_fx_attr else None,
    })
    
    builder.emit(
        f'{acc_next} = "helion.call_torch"(%acc_iter, {lhs_tile}, {rhs_tile}){call_attrs} '
        f": ({ctx.tensor_type}, {ctx.tensor_type}, {ctx.tensor_type}) -> {ctx.tensor_type}"
    )
    
    return acc_next


def _emit_phi_if_present(ctx: LoweringContext) -> None:
    """Emit phi node if present in root FX info."""
    builder = ctx.builder
    
    phi_fx_name = ctx.root_fx_info.get("phi")
    if phi_fx_name is not None:
        phi_result = builder.fresh("phi")
        phi_attrs = format_attr_dict({"fx_node": format_string_attr(phi_fx_name)})
        builder.emit(
            f'{phi_result} = "helion.phi"({ctx.acc_seed}, {ctx.current_acc}){phi_attrs} '
            f": ({ctx.tensor_type}, {ctx.tensor_type}) -> {ctx.tensor_type}"
        )
        ctx.current_acc = phi_result


def _emit_store_tile(ctx: LoweringContext) -> None:
    """Emit store tile operation."""
    builder = ctx.builder
    loop_map = ctx.get_loop_map()
    
    outer_ivs = [loop.iv_name for loop in ctx.outer_loops]
    outer_iv_m = outer_ivs[0] if outer_ivs else "%tile_m_iv"
    outer_iv_n = outer_ivs[1] if len(outer_ivs) > 1 else "%tile_n_iv"
    
    store_tile_m_size = _choose_tile_size(builder, ctx.outer_tile_sizes, loop_map, "tile_m")
    store_tile_n_size = _choose_tile_size(builder, ctx.outer_tile_sizes, loop_map, "tile_n")
    store_meta = format_dynamic_tensor_meta(store_tile_m_size, store_tile_n_size, ctx.element_type)
    
    store_attrs = format_attr_dict({
        "tile": format_indices_attr([outer_iv_m, outer_iv_n]),
        "sizes": ctx.tile_shape_attr,
        "tensor_meta": store_meta,
        "fx_node": format_string_attr(ctx.root_fx_info.get("store"))
            if ctx.root_fx_info.get("store") else None,
    })
    
    builder.emit(
        f'"helion.store_tile_dynamic"({ctx.out_value}, {ctx.current_acc}, {store_tile_m_size}, {store_tile_n_size}){store_attrs} '
        f": ({ctx.tensor_type}, {ctx.tensor_type}, index, index) -> ()"
    )


# -----------------------------------------------------------------------------
# Helper functions for tile size computation
# -----------------------------------------------------------------------------


def _emit_dynamic_tile_size(
    builder: MLIRBuilder,
    iv: str,
    dim: str,
    tile_size: int,
    hint: str,
) -> str:
    """Emit a dynamic tile size computation for concrete tile sizes."""
    return builder.emit_affine_min(
        f"(d0)[s0] -> ({tile_size}, s0 - d0 * {tile_size})",
        [iv],
        [dim],
    )


def _emit_dynamic_tile_size_symbolic(
    builder: MLIRBuilder,
    iv: str,
    dim: str,
    tile_size_ssa: str,
    hint: str,
) -> str:
    """Emit a dynamic tile size computation for symbolic tile sizes."""
    return builder.emit_affine_min(
        "(d0)[s0, s1] -> (s1, s0 - d0 * s1)",
        [iv],
        [dim, tile_size_ssa],
    )


def _choose_tile_size(
    builder: MLIRBuilder,
    dynamic_sizes: dict[str, str],
    loop_map: dict[str, LoopInfo],
    key: str,
) -> str:
    """Choose the appropriate tile size SSA value."""
    if key in dynamic_sizes:
        return dynamic_sizes[key]
    
    fallback_loop = loop_map.get(key)
    if fallback_loop:
        if fallback_loop.tile_const:
            return fallback_loop.tile_const
        if fallback_loop.tile_size is not None:
            const = builder.emit_index_constant(fallback_loop.tile_size)
            fallback_loop.tile_const = const
            return const
    
    return builder.emit_index_constant(0)


# -----------------------------------------------------------------------------
# FX graph inspection helpers
# -----------------------------------------------------------------------------


def _first_graph_with_block_ids(device_ir: "DeviceIR") -> "GraphInfo | None":
    """Find the first graph with block_ids (usually a ForLoopGraphInfo)."""
    for graph_info in device_ir.graphs:
        if getattr(graph_info, "block_ids", None):
            return graph_info
    return None


def _first_root_graph(device_ir: "DeviceIR") -> "GraphInfo | None":
    """Find the first root graph (without block_ids)."""
    for graph_info in device_ir.graphs:
        if getattr(graph_info, "block_ids", None):
            continue
        return graph_info
    return None


def _extract_fx_names(graph_info: "GraphInfo | None") -> dict[str, str]:
    """Extract FX node names for loads and addmm from a graph."""
    if graph_info is None:
        return {}
    
    names: dict[str, str] = {}
    load_seen = 0
    
    for node in graph_info.graph.nodes:
        if node.op == "call_function":
            if node.target is hl_memory_ops.load:
                key = "lhs_load" if load_seen == 0 else "rhs_load"
                names[key] = node.name
                load_seen += 1
            elif node.target is aten.addmm.default:
                names["addmm"] = node.name
    
    return names


def _extract_root_fx_info(graph_info: "GraphInfo | None") -> dict[str, str]:
    """Extract FX node info from root graph (store, phi)."""
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


# -----------------------------------------------------------------------------
# Validation utility
# -----------------------------------------------------------------------------


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
