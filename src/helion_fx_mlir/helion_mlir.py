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
    LoadInfo,
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
    tensor_args = ctx.get_tensor_args()
    
    # Compute output shape from loop extents or first tensor shape
    output_shape = _derive_output_shape(ctx, fake_args, tensor_args)
    ctx.output_shape = output_shape
    full_shape_attr = format_shape_attr(output_shape)
    
    # Emit module start with tile size attributes
    module_attrs = ctx.get_module_attributes()
    builder.emit_module_start(module_attrs)
    
    # Build function signature from kernel arguments only (no tile size args)
    func_args = ctx.get_func_signature_args()
    symbolic_tile_args = ctx.get_symbolic_tile_args()
    
    # Determine result type from output shape
    result_type = format_tensor_type(output_shape, ctx.element_type)
    
    # Emit function start
    builder.emit_func_start(kernel_name, func_args, result_type)
    
    # Emit get_module_attribute ops for each symbolic tile size
    # Map loop names to dimension letters for block_* naming
    def loop_name_to_dim(name: str) -> str:
        if name in {"tile_m", "m"}:
            return "m"
        elif name in {"tile_n", "n"}:
            return "n"
        elif name in {"tile_k", "k"}:
            return "k"
        elif name in {"tile_b", "b"}:
            return "b"
        return name.replace("tile_", "")
    
    for sym_arg in symbolic_tile_args:
        loop_name = sym_arg["name"]
        dim_letter = loop_name_to_dim(loop_name)
        attr_name = f"loom.block_{dim_letter}"
        ssa = builder.emit_get_module_attribute(attr_name, f"block_{dim_letter}")
        ctx.symbolic_arg_ssa[loop_name] = ssa
        builder.emit_comment(f"Block size from module attribute: {attr_name} (block_id={sym_arg['block_id']})")
    
    # Emit output allocation (using first tensor arg as template)
    first_tensor_ssa = ctx.get_tensor_arg_ssa(0) if tensor_args else "%arg0"
    first_tensor_type = tensor_args[0].mlir_type if tensor_args else ctx.tensor_type
    
    ctx.out_value = builder.fresh("out")
    output_type = format_tensor_type(output_shape, ctx.element_type)
    alloc_attrs = format_attr_dict({"shape": full_shape_attr})
    builder.emit(
        f'{ctx.out_value} = "helion.alloc_like"({first_tensor_ssa}){alloc_attrs} : ({first_tensor_type}) -> {output_type}'
    )
    
    # Emit accumulator initialization
    ctx.acc_seed = builder.fresh("acc_init")
    zero_attrs = format_attr_dict({"shape": ctx.tile_shape_attr, "dtype": ctx.element_type})
    builder.emit(
        f'{ctx.acc_seed} = "helion.zero_tile"(){zero_attrs} : () -> {ctx.tensor_type}'
    )
    
    # Emit dimension queries
    _emit_dimension_queries(ctx)
    

    # Set up dimension mapping
    ctx.setup_dims_map()
    
    # Emit loop bounds computation and metadata
    _emit_loop_bounds(ctx)
    
    # Emit outer parallel loop
    _emit_parallel_loop_structure(ctx, bound_kernel)
    
    # Emit function end and return
    builder.emit(f"return {ctx.out_value} : {output_type}")
    builder.emit_func_end()
    builder.emit_module_end()
    
    return builder.build()


def _derive_output_shape(
    ctx: LoweringContext,
    fake_args: tuple,
    tensor_args: list[KernelArgInfo],
) -> list[int | None]:
    """Derive the output shape from tensor shapes or loop extents.
    
    Strategy:
    1. For matmul pattern (2 outer loops, no tile_b): use loop extents [M, N]
    2. For higher-dimensional kernels or when tile_b present: use first input's shape
    """
    import torch
    
    # Check for matmul pattern: no tile_b and exactly 2 outer loops (tile_m, tile_n)
    loop_names = {loop.name for loop in ctx.outer_loops}
    has_batch_dim = "tile_b" in loop_names
    
    # For matmul pattern (tile_m, tile_n only), use loop extents
    if not has_batch_dim and len(ctx.outer_loops) == 2:
        m_extent = ctx.loop_extents.get("tile_m")
        n_extent = ctx.loop_extents.get("tile_n")
        if m_extent is not None and n_extent is not None:
            return [m_extent, n_extent]
    
    # For other patterns (e.g., attention with tile_b, tile_m), 
    # use the first input tensor's shape as output shape
    if tensor_args and tensor_args[0].index < len(fake_args):
        tensor = fake_args[tensor_args[0].index]
        if hasattr(tensor, 'shape'):
            return [int(s) if not isinstance(s, torch.SymInt) else None 
                    for s in tensor.shape]
    
    # Fallback: use loop extents
    output_dims: list[int | None] = []
    for loop in ctx.outer_loops:
        if loop.total_extent is not None:
            output_dims.append(loop.total_extent)
    if len(output_dims) >= 2:
        return output_dims
    
    return [None, None]


def _emit_dimension_queries(ctx: LoweringContext) -> None:
    """Emit dimension values as arith.constant or store as int if known."""
    builder = ctx.builder
    
    # Get concrete dimension values from loop info
    # The total_extent in each loop IS the dimension size
    loop_map = ctx.get_loop_map()
    
    # Get M dimension from tile_m loop
    tile_m_loop = loop_map.get("tile_m")
    if tile_m_loop:
        # If extent is concrete (int), store it directly. Otherwise emit constant.
        # But wait, BoundKernel extents are usually integers even for dynamic shapes 
        # unless they are SymInts.
        # The goal is: if we KNOW it's a fixed integer at compile time (from BoundKernel),
        # we can just use that integer in affine maps.
        
        # However, for true dynamic shapes support in the future, we might want to read from
        # the tensor dims. But for now, we follow the user request to inline the values
        # found in the BoundKernel (which are concrete for the bound arguments).
        
        if isinstance(tile_m_loop.total_extent, int):
            ctx.dim_m = tile_m_loop.total_extent
        else:
            # Fallback for symbolic (though current BoundKernel usually resolves to int or SymInt)
            # If it's SymInt, we probably still want to use it as a symbol or inline it 
            # if the MLIR builder supports it. 
            # For now, let's treat non-ints as requiring SSA values (which we don't have for dims easily yet
            # without reading from tensor).
            # But wait, previous code did `builder.emit_index_constant(tile_m_loop.total_extent)`.
            # If total_extent is a SymInt, emit_index_constant might fail or produce a constant op?
            # Actually, `emit_index_constant` expects an int.
            # So `total_extent` must be int compatible here.
            ctx.dim_m = int(tile_m_loop.total_extent)
    else:
        ctx.dim_m = 0
    
    # Get K dimension from tile_k loop
    tile_k_loop = loop_map.get("tile_k")
    if tile_k_loop:
        ctx.dim_k = int(tile_k_loop.total_extent)
    else:
        ctx.dim_k = 0
    
    # Get N dimension from tile_n loop
    tile_n_loop = loop_map.get("tile_n")
    if tile_n_loop:
        ctx.dim_n = int(tile_n_loop.total_extent)
    else:
        ctx.dim_n = 0


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
                if isinstance(loop_dim, int):
                    # Inline loop_dim
                    trip_count_ssa = builder.emit_affine_apply(
                        f"()[s0] -> ({loop_dim} ceildiv s0)",
                        [],
                        [tile_ssa],
                    )
                else:
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
            # Emit tile size constant - still useful for other things or inline?
            # User wants to avoid arith.constant for SHAPES. 
            # Tile sizes are config parameters, usually constant is fine, but can also be inlined.
            # Let's keep tile size as constant SSA for reusable clarity, or inline it if requested?
            # The prompt said "shapes of inputs ... use directly the value hard-coded inline".
            # It didn't explicitly forbid tile size constants, but consistently inlining everything is better.
            
            # However, `loop.tile_const` is expected to be an SSA value by other parts (e.g. `_emit_dynamic_tile_size`).
            # Let's create the constant for tile size as before, but inline the DIMENSION.
            
            tile_const = builder.fresh(f"{loop.name}_tile")
            builder.emit(f"{tile_const} = arith.constant {loop.tile_size} : index")
            loop.tile_const = tile_const
            
            # Compute trip count with ceildiv
            if isinstance(loop_dim, int):
                # Inline loop_dim
                trip_count_ssa = builder.emit_affine_apply(
                    f"()[s0] -> ({loop_dim} ceildiv s0)",
                    [],
                    [tile_const],
                )
            else:
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
    
    # Emit placeholder SSA values for intermediate tensors
    _emit_intermediate_tensors(ctx, bound_kernel)
    
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
    if for_graph is not None:
        load_infos, other_names = _extract_load_infos(for_graph, ctx.kernel_args)
        ctx.load_infos = load_infos
        ctx.fx_names = other_names
    else:
        ctx.load_infos = []
        ctx.fx_names = {}
    
    root_graph = _first_root_graph(device_ir)
    ctx.root_fx_info = _extract_root_fx_info(root_graph)


def _emit_intermediate_tensors(ctx: LoweringContext, bound_kernel: "BoundKernel") -> None:
    """Emit placeholder SSA values for intermediate tensors.
    
    This handles tensors created via reshape/transpose/view that are 
    referenced in load operations but are not function arguments.
    """
    builder = ctx.builder
    
    # Collect all tensor names that are function arguments
    arg_names = {arg.name for arg in ctx.kernel_args if arg.is_tensor}
    
    # Find load infos that reference non-argument tensors
    for load_info in ctx.load_infos:
        source_name = load_info.source_tensor_name
        
        # Skip if this is already a function argument
        if source_name in arg_names:
            continue
        
        # Skip if we've already emitted this tensor
        if f"%{source_name}" in ctx.fx_value_map.values():
            continue
        
        # Emit a placeholder tensor operation for this intermediate tensor
        # In a future version, we could trace back to the actual operations
        ssa = builder.fresh(source_name)
        placeholder_attrs = format_attr_dict({
            "tensor_name": format_string_attr(source_name),
            "note": format_string_attr("intermediate tensor placeholder"),
        })
        builder.emit(
            f'{ssa} = "helion.intermediate_tensor"(){placeholder_attrs} : () -> {ctx.tensor_type}'
        )
        
        # Store the SSA value for use in loads
        ctx.fx_value_map[source_name] = ssa


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
        
        # Build IV map for loads
        loop_ivs = {
            "tile_m": outer_iv_m,
            "tile_n": outer_iv_n,
            "tile_k": reduction_iv,
        }
        
        # Emit all loads dynamically
        emitted_loads: list[str] = []
        for load_info in ctx.load_infos:
            load_ssa = _emit_load(ctx, load_info, loop_ivs, tile_k_size)
            emitted_loads.append(load_ssa)
            load_info.ssa_name = load_ssa
        
        # Emit computation (addmm) - use first two loads for now
        # TODO: Generalize for other computation patterns
        if len(emitted_loads) >= 2:
            acc_next = _emit_addmm(ctx, emitted_loads[0], emitted_loads[1])
        elif len(emitted_loads) == 1:
            # Single load case - just pass through
            acc_next = emitted_loads[0]
        else:
            acc_next = "%acc_iter"
        
        # Emit yield
        builder.emit(f"affine.yield {acc_next} : {ctx.tensor_type}")
        builder.pop()
        builder.emit("}")
        
        ctx.current_acc = loop_result


def _compute_reduction_trip_bound(
    ctx: LoweringContext, loop: LoopInfo, loop_dim: str | int | None
) -> str:
    """Compute the trip bound for a reduction loop."""
    builder = ctx.builder
    
    if loop.is_symbolic:
        tile_ssa = ctx.symbolic_arg_ssa.get(loop.name)
        if tile_ssa is None:
            raise ValueError(f"Missing symbolic argument for reduction loop {loop.name}")
        loop.tile_const = tile_ssa
        
        if loop_dim is not None:
            if isinstance(loop_dim, int):
                trip_count_ssa = builder.emit_affine_apply(
                    f"()[s0] -> ({loop_dim} ceildiv s0)",
                    [],
                    [tile_ssa],
                )
            else:
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
        
        if isinstance(loop_dim, int):
            trip_count_ssa = builder.emit_affine_apply(
                f"()[s0] -> ({loop_dim} ceildiv s0)",
                [],
                [tile_const],
            )
        else:
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
    ctx: LoweringContext, loop: LoopInfo, loop_dim: str | int | None
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


def _emit_load(
    ctx: LoweringContext,
    load_info: LoadInfo,
    loop_ivs: dict[str, str],
    reduction_tile_size: str,
) -> str:
    """Emit a tile load operation for any input tensor.
    
    This generalizes the previous _emit_lhs_load and _emit_rhs_load functions
    to handle any number of input tensors by extracting info from LoadInfo.
    
    Args:
        ctx: Lowering context
        load_info: Information about this load extracted from FX graph
        loop_ivs: Map from loop name to IV SSA value (e.g., {"tile_m": "%tile_m_iv"})
        reduction_tile_size: SSA value for the reduction tile size (tile_k)
    
    Returns:
        SSA name of the loaded tile
    """
    builder = ctx.builder
    loop_map = ctx.get_loop_map()
    
    # Get tensor info from kernel args
    tensor_arg = _get_tensor_arg_for_load(ctx, load_info)
    
    # Determine tensor SSA - check fx_value_map for intermediate tensors first
    if tensor_arg:
        tensor_ssa = tensor_arg.ssa_name
        tensor_type = tensor_arg.mlir_type
    elif load_info.source_tensor_name in ctx.fx_value_map:
        tensor_ssa = ctx.fx_value_map[load_info.source_tensor_name]
        tensor_type = ctx.tensor_type
    else:
        tensor_ssa = f"%{load_info.source_tensor_name}"
        tensor_type = ctx.tensor_type
    
    # Determine tile dimensions based on source tensor
    # For matmul pattern: LHS is [M, K], RHS is [K, N]
    tile_dims = _infer_tile_dims_for_load(ctx, load_info, tensor_arg)
    
    # Get tile sizes for each dimension
    tile_sizes: list[str] = []
    indices: list[str] = []
    for dim_name in tile_dims:
        if dim_name == "tile_k":
            tile_sizes.append(reduction_tile_size)
            indices.append(loop_ivs.get("tile_k", "%tile_k_iv"))
        else:
            tile_size = _choose_tile_size(builder, ctx.outer_tile_sizes, loop_map, dim_name)
            tile_sizes.append(tile_size)
            indices.append(loop_ivs.get(dim_name, f"%{dim_name}_iv"))
    
    # Ensure we have exactly 2 dimensions for 2D tensors
    while len(tile_sizes) < 2:
        tile_sizes.append(reduction_tile_size)
        indices.append(loop_ivs.get("tile_k", "%tile_k_iv"))
    
    # Create fresh SSA name for the load result
    load_result = builder.fresh(load_info.source_tensor_name)
    
    # Format attributes
    indices_attr = format_indices_attr(indices[:2])
    tensor_meta = format_dynamic_tensor_meta(tile_sizes[0], tile_sizes[1], ctx.element_type)
    
    attrs = format_attr_dict({
        "tile": indices_attr,
        "sizes": ctx.tile_shape_attr,
        "tensor_meta": tensor_meta,
        "fx_node": format_string_attr(load_info.fx_node_name) if load_info.fx_node_name else None,
    })
    
    builder.emit(
        f'{load_result} = "helion.load_tile_dynamic"({tensor_ssa}, {tile_sizes[0]}, {tile_sizes[1]}){attrs} '
        f": ({tensor_type}, index, index) -> {ctx.tensor_type}"
    )
    
    return load_result


def _get_tensor_arg_for_load(
    ctx: LoweringContext, load_info: LoadInfo
) -> KernelArgInfo | None:
    """Get the kernel argument corresponding to a load's source tensor."""
    # First try by index if available
    if load_info.source_tensor_arg_idx is not None:
        tensor_args = ctx.get_tensor_args()
        if load_info.source_tensor_arg_idx < len(tensor_args):
            return tensor_args[load_info.source_tensor_arg_idx]
    
    # Fall back to matching by name
    return ctx.get_tensor_arg_by_name(load_info.source_tensor_name)


def _infer_tile_dims_for_load(
    ctx: LoweringContext,
    load_info: LoadInfo,
    tensor_arg: KernelArgInfo | None,
) -> list[str]:
    """Infer the tile dimension names for a load operation.
    
    This determines which loop dimensions correspond to which tensor dimensions.
    For matmul: LHS[M,K] -> [tile_m, tile_k], RHS[K,N] -> [tile_k, tile_n]
    """
    # If the LoadInfo already has explicit tile_dim_names, use them
    if load_info.tile_dim_names:
        return load_info.tile_dim_names
    
    # Infer based on tensor position in kernel args
    # For typical matmul: first tensor is LHS [M,K], second is RHS [K,N]
    tensor_args = ctx.get_tensor_args()
    
    if tensor_arg is None:
        # Fallback to order-based inference
        return ["tile_m", "tile_k"]
    
    try:
        idx = tensor_args.index(tensor_arg)
    except ValueError:
        return ["tile_m", "tile_k"]
    
    if idx == 0:
        # First tensor: LHS pattern [M, K]
        return ["tile_m", "tile_k"]
    elif idx == 1:
        # Second tensor: RHS pattern [K, N]
        return ["tile_k", "tile_n"]
    else:
        # Additional tensors: default to [M, K] pattern
        return ["tile_m", "tile_k"]


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
    
    # Get tile sizes for the first two outer loops (may be tile_b, tile_m, etc.)
    first_loop = ctx.outer_loops[0] if ctx.outer_loops else None
    second_loop = ctx.outer_loops[1] if len(ctx.outer_loops) > 1 else None
    
    store_tile_1 = _choose_tile_size(builder, ctx.outer_tile_sizes, loop_map, 
                                      first_loop.name if first_loop else "tile_m")
    store_tile_2 = _choose_tile_size(builder, ctx.outer_tile_sizes, loop_map, 
                                      second_loop.name if second_loop else "tile_n")
    store_meta = format_dynamic_tensor_meta(store_tile_1, store_tile_2, ctx.element_type)
    
    store_attrs = format_attr_dict({
        "tile": format_indices_attr([outer_iv_m, outer_iv_n]),
        "sizes": ctx.tile_shape_attr,
        "tensor_meta": store_meta,
        "fx_node": format_string_attr(ctx.root_fx_info.get("store"))
            if ctx.root_fx_info.get("store") else None,
    })
    
    # Use stored output_shape for consistent tensor type
    output_type = format_tensor_type(ctx.output_shape, ctx.element_type)
    
    builder.emit(
        f'"helion.store_tile_dynamic"({ctx.out_value}, {ctx.current_acc}, {store_tile_1}, {store_tile_2}){store_attrs} '
        f": ({output_type}, {ctx.tensor_type}, index, index) -> ()"
    )


# -----------------------------------------------------------------------------
# Helper functions for tile size computation
# -----------------------------------------------------------------------------


def _emit_dynamic_tile_size(
    builder: MLIRBuilder,
    iv: str,
    dim: str | int,
    tile_size: int,
    hint: str,
) -> str:
    """Emit a dynamic tile size computation for concrete tile sizes.
    
    If dim is an integer, it is inlined directly into the affine expression.
    If dim is a string SSA value, it is passed as a symbol.
    """
    if isinstance(dim, int):
        # Inline the integer dimension directly
        return builder.emit_affine_min(
            f"(d0) -> ({tile_size}, {dim} - d0 * {tile_size})",
            [iv],
            [],  # No symbols needed when dim is inlined
        )
    else:
        return builder.emit_affine_min(
            f"(d0)[s0] -> ({tile_size}, s0 - d0 * {tile_size})",
            [iv],
            [dim],
        )


def _emit_dynamic_tile_size_symbolic(
    builder: MLIRBuilder,
    iv: str,
    dim: str | int,
    tile_size_ssa: str,
    hint: str,
) -> str:
    """Emit a dynamic tile size computation for symbolic tile sizes."""
    if isinstance(dim, int):
        return builder.emit_affine_min(
            f"(d0)[s0] -> (s0, {dim} - d0 * s0)",
            [iv],
            [tile_size_ssa],
        )
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


def _extract_load_infos(
    graph_info: "GraphInfo | None",
    kernel_args: list[KernelArgInfo],
) -> tuple[list[LoadInfo], dict[str, str]]:
    """Extract load operation information and other FX names from a graph.
    
    Returns:
        A tuple of (list of LoadInfo, dict of other FX names like addmm).
    """
    if graph_info is None:
        return [], {}
    
    load_infos: list[LoadInfo] = []
    other_names: dict[str, str] = {}
    
    # Build a map from tensor name to kernel arg index
    tensor_name_to_idx: dict[str, int] = {}
    for arg in kernel_args:
        if arg.is_tensor:
            tensor_name_to_idx[arg.name] = arg.index
    
    for node in graph_info.graph.nodes:
        if node.op == "call_function":
            if node.target is hl_memory_ops.load:
                # Extract source tensor from first argument
                tensor_arg = node.args[0] if node.args else None
                source_name = _get_source_tensor_name(tensor_arg)
                
                # Match to kernel argument
                arg_idx = tensor_name_to_idx.get(source_name)
                
                # Extract tile dimension names from args[1] if present
                tile_dim_names = _parse_tile_indices(node.args[1] if len(node.args) > 1 else None)
                
                load_infos.append(LoadInfo(
                    fx_node_name=node.name,
                    source_tensor_name=source_name,
                    source_tensor_arg_idx=arg_idx,
                    tile_dim_names=tile_dim_names,
                ))
            elif node.target is aten.addmm.default:
                other_names["addmm"] = node.name
    
    return load_infos, other_names


def _parse_tile_indices(indices_arg: object) -> list[str]:
    """Parse tile index expression to extract dimension names.
    
    The indices_arg is typically a string like "[sym_size_int, block_size_2]"
    or a list of FX nodes representing tile dimensions.
    
    Returns list of dimension names like ["tile_m", "tile_k"].
    """
    import torch.fx
    import re
    
    if indices_arg is None:
        return []
    
    # If it's a list of FX nodes, extract their names
    if isinstance(indices_arg, (list, tuple)):
        names = []
        for item in indices_arg:
            if isinstance(item, torch.fx.Node):
                # Convert node name to tile dimension name
                name = _symnode_to_tile_dim(item.name)
                names.append(name)
            elif hasattr(item, 'name'):
                names.append(_symnode_to_tile_dim(item.name))
        return names
    
    # If it's a string, try to parse it
    if isinstance(indices_arg, str):
        # Extract names from "[name1, name2]" format
        # Or "TileIndex([name1, name2])" format
        match = re.search(r'\[([^\]]+)\]', indices_arg)
        if match:
            inner = match.group(1)
            parts = [p.strip() for p in inner.split(',')]
            return [_symnode_to_tile_dim(p) for p in parts if p]
    
    return []


def _symnode_to_tile_dim(name: str) -> str:
    """Convert a SymNode name to a tile dimension name.
    
    E.g., "sym_size_int" -> "tile_m", "block_size_2" -> "tile_k"
    Or preserve known names like "tile_m", "tile_n", "tile_k".
    """
    # Already a tile name
    if name.startswith("tile_"):
        return name
    
    # Common SymNode patterns from Helion
    # sym_size_int typically refers to first dim (M), sym_size_int_1 to second (N)
    if "block_size" in name or "sym_size" in name:
        # Extract the index if present
        import re
        match = re.search(r'(\d+)$', name)
        if match:
            idx = int(match.group(1))
            # Map block_size indices to tile names
            # block_size_0 -> tile_m, block_size_1 -> tile_n, block_size_2 -> tile_k
            dim_map = {0: "tile_m", 1: "tile_n", 2: "tile_k", 3: "tile_b"}
            return dim_map.get(idx, f"tile_{idx}")
        else:
            # No index, assume first dimension
            return "tile_m"
    
    # Fallback: just prefix with tile_
    return f"tile_{name}"


def _get_source_tensor_name(tensor_arg: object) -> str:
    """Extract the source tensor name from an FX node argument.
    
    The tensor argument in a load node is typically a node representing
    the _host_tensor('name') call, so we extract the name from it.
    """
    import torch.fx
    
    if tensor_arg is None:
        return "unknown"
    
    if isinstance(tensor_arg, torch.fx.Node):
        # The node name itself is often the tensor name (e.g., "x", "y")
        return tensor_arg.name
    
    return str(tensor_arg)


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
