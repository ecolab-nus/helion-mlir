"""Lowering context for FX-to-MLIR conversion.

This module defines the LoweringContext dataclass that holds all state
needed during the lowering process, including the MLIR builder, kernel
metadata, and mappings between FX nodes and MLIR SSA values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

from .mlir_builder import (
    MLIRBuilder,
    is_concrete_size,
    torch_dtype_to_mlir_element_type,
    format_tensor_type,
)

if TYPE_CHECKING:
    from helion._compiler.device_ir import DeviceIR, GraphInfo
    from helion._compiler.compile_environment import BlockSizeInfo
    from helion.runtime.kernel import BoundKernel
    from torch import Tensor
    import torch


@dataclass
class LoopInfo:
    """Information about a loop in the MLIR emission."""
    
    block_id: int
    name: str
    tile_size: int | None  # None for symbolic
    trip_count: int | None  # None for symbolic
    total_extent: int
    is_symbolic: bool
    tile_const: str | None = None  # SSA value for tile size constant
    trip_count_ssa: str | None = None  # SSA value for trip count
    iv_name: str | None = None  # Induction variable SSA name


@dataclass
class KernelArgInfo:
    """Information about a kernel function argument."""
    
    name: str  # Original parameter name from kernel signature
    index: int  # Position in the argument list
    is_tensor: bool  # Whether this is a tensor argument
    dtype: "torch.dtype | None" = None  # Tensor dtype if is_tensor
    shape: list[int | None] | None = None  # Tensor shape (None for dynamic dims)
    mlir_type: str | None = None  # MLIR type string
    ssa_name: str | None = None  # SSA value name (e.g., "%arg0")


@dataclass
class LoweringContext:
    """Context passed to lowering functions during FX-to-MLIR conversion.
    
    This class holds all the state needed during the lowering process,
    including references to the builder, kernel metadata, and mappings
    between FX values and MLIR SSA values.
    """
    
    # Core builder
    builder: MLIRBuilder
    
    # Kernel metadata
    bound_kernel: Any  # BoundKernel, using Any to avoid import issues
    device_ir: Any  # DeviceIR
    
    # Current graph being processed
    current_graph: Any | None = None  # GraphInfo
    
    # Type information
    element_type: str = "f32"
    tensor_type: str = "tensor<?x?xf32>"
    
    # Dimension SSA values
    dim_m: str | None = None
    dim_n: str | None = None
    dim_k: str | None = None
    
    # Output tensor
    out_value: str | None = None
    acc_seed: str | None = None
    
    # Loop information
    outer_loops: list[LoopInfo] = field(default_factory=list)
    reduction_loops: list[LoopInfo] = field(default_factory=list)
    
    # Symbolic tile size arguments (name -> SSA value)
    symbolic_arg_ssa: dict[str, str] = field(default_factory=dict)
    
    # Dynamic tile sizes computed inside loops (name -> SSA value)
    outer_tile_sizes: dict[str, str] = field(default_factory=dict)
    
    # Dimension to SSA value mapping
    dims_map: dict[str, str] = field(default_factory=dict)
    
    # FX node name to MLIR SSA value mapping
    fx_value_map: dict[str, str] = field(default_factory=dict)
    
    # Block sizes from the kernel environment
    block_sizes: dict[int, Any] = field(default_factory=dict)  # BlockSizeInfo
    
    # Current accumulator value in reduction loops
    current_acc: str | None = None
    
    # Grid block IDs for parallel loops
    parallel_block_ids: list[int] = field(default_factory=list)
    
    # Extracted FX node names for annotation
    fx_names: dict[str, str] = field(default_factory=dict)
    root_fx_info: dict[str, str] = field(default_factory=dict)
    
    # Tile shape attribute string
    tile_shape_attr: str = "[]"
    
    # Kernel function arguments (extracted from bound kernel)
    kernel_args: list[KernelArgInfo] = field(default_factory=list)
    
    @classmethod
    def from_bound_kernel(
        cls,
        bound_kernel: "BoundKernel",
        kernel_name: str = "helion_kernel",
    ) -> "LoweringContext":
        """Create a LoweringContext from a bound Helion kernel.
        
        This factory method extracts all necessary metadata from the kernel
        and initializes the context for MLIR emission.
        """
        import torch
        
        builder = MLIRBuilder()
        
        fake_args = bound_kernel.fake_args
        if len(fake_args) < 2:
            raise ValueError("Expected the kernel to expose at least two tensor arguments.")
        
        # Extract kernel argument information from the signature
        kernel_args = _extract_kernel_args(bound_kernel)
        
        # Find the first two tensor arguments for dimension inference
        tensor_args = [arg for arg in kernel_args if arg.is_tensor]
        if len(tensor_args) < 2:
            raise ValueError("Expected at least two tensor arguments.")
        
        lhs_info = tensor_args[0]
        rhs_info = tensor_args[1]
        lhs = fake_args[lhs_info.index]
        rhs = fake_args[rhs_info.index]
        
        # Get block sizes
        block_sizes = {info.block_id: info for info in bound_kernel.env.block_sizes}
        
        # Get grid block IDs
        device_ir = bound_kernel.host_function.device_ir
        grid_block_groups: Sequence[Sequence[int]] = device_ir.grid_block_ids
        if not grid_block_groups:
            raise ValueError("device_ir.grid_block_ids is empty; nothing to lower.")
        parallel_block_ids = list(grid_block_groups[0])
        
        # Determine element type from first tensor
        element_type = torch_dtype_to_mlir_element_type(lhs.dtype)
        tensor_type = format_tensor_type([None, None], element_type)
        
        # Create context
        ctx = cls(
            builder=builder,
            bound_kernel=bound_kernel,
            device_ir=device_ir,
            element_type=element_type,
            tensor_type=tensor_type,
            block_sizes=block_sizes,
            parallel_block_ids=parallel_block_ids,
            kernel_args=kernel_args,
        )
        
        # Build loop information
        ctx._build_loop_info(lhs, rhs)
        
        return ctx
    
    def get_tensor_args(self) -> list[KernelArgInfo]:
        """Get only the tensor arguments from kernel_args."""
        return [arg for arg in self.kernel_args if arg.is_tensor]
    
    def get_func_signature_args(self) -> list[tuple[str, str]]:
        """Get function signature arguments as (ssa_name, mlir_type) tuples.
        
        Returns only kernel arguments. Symbolic tile sizes are now module attributes.
        """
        args = []
        
        # Add kernel arguments only
        for arg in self.kernel_args:
            if arg.ssa_name and arg.mlir_type:
                args.append((arg.ssa_name, arg.mlir_type))
        
        return args
    
    def get_lhs_tensor_ssa(self) -> str:
        """Get SSA name of the first (LHS) tensor argument."""
        tensor_args = self.get_tensor_args()
        if tensor_args:
            return tensor_args[0].ssa_name or "%arg0"
        return "%arg0"
    
    def get_rhs_tensor_ssa(self) -> str:
        """Get SSA name of the second (RHS) tensor argument."""
        tensor_args = self.get_tensor_args()
        if len(tensor_args) > 1:
            return tensor_args[1].ssa_name or "%arg1"
        return "%arg1"
    
    def get_lhs_tensor_type(self) -> str:
        """Get MLIR type of the first (LHS) tensor argument."""
        tensor_args = self.get_tensor_args()
        if tensor_args:
            return tensor_args[0].mlir_type or self.tensor_type
        return self.tensor_type
    
    def get_rhs_tensor_type(self) -> str:
        """Get MLIR type of the second (RHS) tensor argument."""
        tensor_args = self.get_tensor_args()
        if len(tensor_args) > 1:
            return tensor_args[1].mlir_type or self.tensor_type
        return self.tensor_type
    
    def get_tensor_arg_by_name(self, name: str) -> KernelArgInfo | None:
        """Get a kernel argument by name."""
        for arg in self.kernel_args:
            if arg.name == name:
                return arg
        return None
    
    def get_tensor_arg_ssa(self, index: int) -> str:
        """Get SSA name of tensor argument at given index."""
        tensor_args = self.get_tensor_args()
        if index < len(tensor_args):
            return tensor_args[index].ssa_name or f"%arg{index}"
        return f"%arg{index}"
    
    def _build_loop_info(self, lhs: "Tensor", rhs: "Tensor") -> None:
        """Build loop information from block sizes and parallel block IDs."""
        symbolic_tile_args: list[dict[str, object]] = []
        
        # Build outer (parallel) loop info
        for block_id in self.parallel_block_ids:
            info = self.block_sizes[block_id]
            block_name = first_debug_name(info.debug_names, fallback=f"block_{block_id}")
            total_extent = resolve_extent(block_name, lhs, rhs)
            
            if is_concrete_size(info.size):
                tile_size = int(info.size)
                trip_count = max(1, math.ceil(total_extent / tile_size))
                is_symbolic = False
            else:
                tile_size = None
                trip_count = None
                is_symbolic = True
                symbolic_tile_args.append({
                    "block_id": block_id,
                    "name": block_name,
                    "arg_name": f"{block_name}_size",
                })
            
            self.outer_loops.append(LoopInfo(
                block_id=block_id,
                name=block_name,
                tile_size=tile_size,
                trip_count=trip_count,
                total_extent=total_extent,
                is_symbolic=is_symbolic,
            ))
        
        # Build reduction loop info
        reduction_block_ids = collect_reduction_block_ids(self.device_ir)
        reduction_block_ids = [
            bid for bid in reduction_block_ids if bid not in self.parallel_block_ids
        ]
        
        for block_id in reduction_block_ids:
            info = self.block_sizes[block_id]
            block_name = first_debug_name(info.debug_names, fallback=f"block_{block_id}")
            total_extent = resolve_extent(block_name, lhs, rhs)
            
            if is_concrete_size(info.size):
                tile_size = int(info.size)
                trip_count = max(1, math.ceil(total_extent / tile_size))
                is_symbolic = False
            else:
                tile_size = None
                trip_count = None
                is_symbolic = True
                if not any(arg["name"] == block_name for arg in symbolic_tile_args):
                    symbolic_tile_args.append({
                        "block_id": block_id,
                        "name": block_name,
                        "arg_name": f"{block_name}_size",
                    })
            
            self.reduction_loops.append(LoopInfo(
                block_id=block_id,
                name=block_name,
                tile_size=tile_size,
                trip_count=trip_count,
                total_extent=total_extent,
                is_symbolic=is_symbolic,
            ))
        
        # Store symbolic arguments for later
        self._symbolic_tile_args = symbolic_tile_args
        
        # Build tile shape attribute
        tile_shape = []
        for loop in self.outer_loops[:2]:
            if loop.is_symbolic:
                tile_shape.append(None)
            else:
                tile_shape.append(loop.tile_size)
        from .mlir_builder import format_shape_attr
        self.tile_shape_attr = format_shape_attr(tile_shape)
    
    def get_symbolic_tile_args(self) -> list[dict[str, object]]:
        """Get the list of symbolic tile size arguments."""
        return getattr(self, "_symbolic_tile_args", [])
    
    def get_module_attributes(self) -> dict[str, tuple[int, str]]:
        """Get tile sizes as module attributes.
        
        Returns a dict mapping attribute name to (value, type) for module attributes.
        Uses a default value (64) for symbolic tile sizes.
        """
        attrs = {}
        all_loops = self.outer_loops + self.reduction_loops
        
        for loop in all_loops:
            attr_name = f"loom.{loop.name}"
            if loop.is_symbolic:
                # Use a placeholder value for symbolic sizes
                attrs[attr_name] = (64, "index")
            elif loop.tile_size is not None:
                attrs[attr_name] = (loop.tile_size, "index")
        
        return attrs
    
    def get_loop_map(self) -> dict[str, LoopInfo]:
        """Get a mapping from loop name to LoopInfo."""
        loop_map = {loop.name: loop for loop in self.outer_loops}
        loop_map.update({loop.name: loop for loop in self.reduction_loops})
        return loop_map
    
    def setup_dims_map(self) -> None:
        """Set up the dimension name to SSA value mapping."""
        has_tile_b = any(loop.name == "tile_b" for loop in self.outer_loops)
        
        self.dims_map = {
            "tile_n": self.dim_n,
            "n": self.dim_n,
            "tile_k": self.dim_k,
            "k": self.dim_k,
        }
        
        if has_tile_b:
            self.dims_map["tile_b"] = self.dim_m
            self.dims_map["b"] = self.dim_m
            self.dims_map["tile_m"] = self.dim_k
        else:
            self.dims_map["tile_m"] = self.dim_m
            self.dims_map["m"] = self.dim_m


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def _extract_kernel_args(bound_kernel: "BoundKernel") -> list[KernelArgInfo]:
    """Extract kernel argument information from the bound kernel.
    
    This function inspects the kernel signature and fake_args to build
    a list of KernelArgInfo describing each argument.
    """
    import torch
    
    kernel_args = []
    fake_args = bound_kernel.fake_args
    
    # Get parameter names from the kernel signature
    param_names = list(bound_kernel.kernel.signature.parameters.keys())
    
    for idx, (name, fake_arg) in enumerate(zip(param_names, fake_args)):
        if isinstance(fake_arg, torch.Tensor):
            # Tensor argument
            dtype = fake_arg.dtype
            # Use None for dynamic dimensions
            shape = [
                int(s) if not isinstance(s, torch.SymInt) else None
                for s in fake_arg.shape
            ]
            element_type = torch_dtype_to_mlir_element_type(dtype)
            mlir_type = format_tensor_type(shape, element_type)
            
            kernel_args.append(KernelArgInfo(
                name=name,
                index=idx,
                is_tensor=True,
                dtype=dtype,
                shape=shape,
                mlir_type=mlir_type,
                ssa_name=f"%{name}",
            ))
        elif isinstance(fake_arg, (int, float)):
            # Scalar argument
            if isinstance(fake_arg, int):
                mlir_type = "index"  # or "i64" depending on usage
            else:
                mlir_type = "f32"  # or "f64"
            
            kernel_args.append(KernelArgInfo(
                name=name,
                index=idx,
                is_tensor=False,
                mlir_type=mlir_type,
                ssa_name=f"%{name}",
            ))
        else:
            # Other types (constexpr, etc.) - may not need MLIR representation
            kernel_args.append(KernelArgInfo(
                name=name,
                index=idx,
                is_tensor=False,
                mlir_type=None,
                ssa_name=None,
            ))
    
    return kernel_args


def first_debug_name(names: set[str], *, fallback: str) -> str:
    """Get the first debug name from a set, sanitizing for MLIR."""
    for name in names:
        if name:
            return name.replace(".", "_").replace("-", "_")
    return fallback


def resolve_extent(name: str, lhs: "Tensor", rhs: "Tensor") -> int:
    """Resolve the extent for a named dimension."""
    if name in {"tile_m", "m"}:
        return int(lhs.size(0))
    if name in {"tile_b", "b"}:
        return int(lhs.size(0))
    if name in {"tile_n", "n"}:
        return int(rhs.size(1))
    if name in {"tile_k", "k"}:
        return int(lhs.size(1))
    raise ValueError(f"Cannot resolve extent for loop named '{name}'.")


def collect_reduction_block_ids(device_ir: "DeviceIR") -> list[int]:
    """Collect all block IDs that are used in reduction loops."""
    block_ids: list[int] = []
    for graph_info in device_ir.graphs:
        candidate = getattr(graph_info, "block_ids", None)
        if candidate:
            for block_id in candidate:
                if block_id not in block_ids:
                    block_ids.append(block_id)
    return block_ids
