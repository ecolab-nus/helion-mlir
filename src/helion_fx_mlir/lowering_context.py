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
    
    # Type information
    element_type: str = "f32"
    tensor_type: str = "tensor<?x?xf32>"
    
    # Dimension SSA values (or concrete int values) - keyed by loop name
    # e.g., {"tile_m": 128, "tile_n": 256, "tile_k": 128}
    loop_extents: dict[str, int] = field(default_factory=dict)
    
    # Loop information
    outer_loops: list[LoopInfo] = field(default_factory=list)
    reduction_loops: list[LoopInfo] = field(default_factory=list)
    # Block sizes from the kernel environment
    block_sizes: dict[int, Any] = field(default_factory=dict)  # BlockSizeInfo
    
    # Grid block IDs for parallel loops
    parallel_block_ids: list[int] = field(default_factory=list)
    
    # Kernel function arguments (extracted from bound kernel)
    kernel_args: list[KernelArgInfo] = field(default_factory=list)
    
    # Symbol table for _get_symnode values (name -> SSA value)
    symbols: dict[str, str] = field(default_factory=dict)
    
    # Host tensor name to function argument mapping (tensor name -> SSA)
    host_tensors: dict[str, str] = field(default_factory=dict)
    
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
        
        # Extract kernel argument information from the signature
        kernel_args = _extract_kernel_args(bound_kernel)
        
        # Get all tensor arguments (any number, including 0)
        tensor_args = [arg for arg in kernel_args if arg.is_tensor]
        
        # Get block sizes
        block_sizes = {info.block_id: info for info in bound_kernel.env.block_sizes}
        
        # Get grid block IDs
        device_ir = bound_kernel.host_function.device_ir
        grid_block_groups: Sequence[Sequence[int]] = device_ir.grid_block_ids
        if not grid_block_groups:
            raise ValueError("device_ir.grid_block_ids is empty; nothing to lower.")
        parallel_block_ids = list(grid_block_groups[0])
        
        # Determine element type from first tensor argument, fallback to f32
        element_type = "f32"
        if tensor_args:
            first_tensor = fake_args[tensor_args[0].index]
            element_type = torch_dtype_to_mlir_element_type(first_tensor.dtype)
        
        # Create context with placeholder tensor_type
        ctx = cls(
            builder=builder,
            bound_kernel=bound_kernel,
            device_ir=device_ir,
            element_type=element_type,
            tensor_type="",  # Will be set after _build_loop_info
            block_sizes=block_sizes,
            parallel_block_ids=parallel_block_ids,
            kernel_args=kernel_args,
        )
        
        # Build loop information - this populates loop_extents
        ctx._build_loop_info_generic()
        
        # Keep tensor_type dynamic for intermediate tile operations
        ctx.tensor_type = format_tensor_type([None, None], element_type)
        
        # Update kernel_args with actual shapes from fake_args
        for arg in ctx.kernel_args:
            if arg.is_tensor and arg.index < len(fake_args):
                tensor = fake_args[arg.index]
                # Use actual tensor shape
                arg.shape = [int(s) if not isinstance(s, torch.SymInt) else None 
                             for s in tensor.shape]
                arg.mlir_type = format_tensor_type(arg.shape, element_type)
        
        return ctx
    
    def get_tensor_args(self) -> list[KernelArgInfo]:
        """Get only the tensor arguments from kernel_args."""
        return [arg for arg in self.kernel_args if arg.is_tensor]
    
    def _build_loop_info_generic(self) -> None:
        """Build loop information from block sizes without assuming matmul patterns.
        
        This method extracts loop extents from the BlockSizeInfo metadata,
        which contains the total size for each dimension.
        """
        symbolic_tile_args: list[dict[str, object]] = []
        
        # Get tensor arguments for extent resolution
        fake_args = self.bound_kernel.fake_args
        tensor_args = self.get_tensor_args()
        
        # Build outer (parallel) loop info
        for block_id in self.parallel_block_ids:
            info = self.block_sizes[block_id]
            block_name = first_debug_name(info.debug_names, fallback=f"block_{block_id}")
            
            # Get extent from BlockSizeInfo or from tensor dimensions
            total_extent = self._resolve_extent_generic(block_name, info, fake_args, tensor_args)
            
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
            
            # Store extent in loop_extents dict
            self.loop_extents[block_name] = total_extent
        
        # Build reduction loop info
        reduction_block_ids = collect_reduction_block_ids(self.device_ir)
        reduction_block_ids = [
            bid for bid in reduction_block_ids if bid not in self.parallel_block_ids
        ]
        
        for block_id in reduction_block_ids:
            info = self.block_sizes[block_id]
            block_name = first_debug_name(info.debug_names, fallback=f"block_{block_id}")
            
            total_extent = self._resolve_extent_generic(block_name, info, fake_args, tensor_args)
            
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
            
            # Store extent in loop_extents dict
            self.loop_extents[block_name] = total_extent
        
        pass  # All loop info built
    
    def _resolve_extent_generic(
        self,
        block_name: str,
        info: Any,
        fake_args: tuple,
        tensor_args: list[KernelArgInfo],
    ) -> int:
        """Resolve extent for a named dimension without matmul assumptions.
        
        Strategy:
        1. Try to get extent from loop variable symbol if available
        2. Fall back to matmul-style naming conventions for compatibility
        3. Use 1 as default if unknown
        """
        import torch
        
        # Check if info has a start/end range we can extract extent from
        # BlockSizeInfo typically has 'size' and related metadata
        
        # For matmul compatibility, check conventional names
        if block_name in {"tile_m", "m"}:
            # First tensor, first dimension
            if tensor_args and tensor_args[0].index < len(fake_args):
                return int(fake_args[tensor_args[0].index].size(0))
        elif block_name in {"tile_n", "n"}:
            # Second tensor, second dimension (or first tensor if only one)
            if len(tensor_args) > 1 and tensor_args[1].index < len(fake_args):
                return int(fake_args[tensor_args[1].index].size(1))
            elif tensor_args and tensor_args[0].index < len(fake_args):
                return int(fake_args[tensor_args[0].index].size(-1))
        elif block_name in {"tile_k", "k"}:
            # First tensor, second dimension (reduction dim)
            if tensor_args and tensor_args[0].index < len(fake_args):
                return int(fake_args[tensor_args[0].index].size(1))
        elif block_name in {"tile_b", "b"}:
            # Batch dimension
            if tensor_args and tensor_args[0].index < len(fake_args):
                return int(fake_args[tensor_args[0].index].size(0))
        
        # Default: try to match dimension index from block name
        # e.g., "block_0" -> dimension 0
        import re
        match = re.search(r"(\d+)$", block_name)
        if match:
            dim_idx = int(match.group(1))
            if tensor_args and tensor_args[0].index < len(fake_args):
                tensor = fake_args[tensor_args[0].index]
                if dim_idx < tensor.ndim:
                    return int(tensor.size(dim_idx))
        
        # Fallback: return 1 as a safe default
        return 1
    
    def get_module_attributes(self) -> dict[str, tuple[object, str]]:
        """Get block sizes as module attributes.
        
        Returns a dict mapping attribute name to (value, type) for module attributes.
        Uses -1 for undefined (symbolic) block sizes.
        
        Naming convention:
        - loom.block_size_0, loom.block_size_1, ... = tile sizes for each block ID
        """
        attrs = {}
        all_loops = self.outer_loops + self.reduction_loops
        
        # Emit block sizes using block_id-based naming
        for loop in all_loops:
            attr_name = f"loom.block_size_{loop.block_id}"
            if loop.is_symbolic:
                # Use -1 for undefined/symbolic sizes
                attrs[attr_name] = (-1, "index")
            elif loop.tile_size is not None:
                attrs[attr_name] = (loop.tile_size, "index")
            else:
                attrs[attr_name] = (-1, "index")
        
        return attrs
    
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
