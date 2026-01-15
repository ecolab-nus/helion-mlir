"""Lowering context for FX-to-MLIR conversion.

This module defines the LoweringContext dataclass that holds all state
needed during the lowering process, including the MLIR builder, kernel
metadata, and mappings between FX nodes and MLIR SSA values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

from .mlir_builder import (
    MLIRBuilder,
    torch_dtype_to_mlir_element_type,
    format_tensor_type,
)

if TYPE_CHECKING:
    from torch import Tensor
    import torch





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



class LoweringContext:
    """Context passed to lowering functions during FX-to-MLIR conversion.
    
    This class holds all the state needed during the lowering process,
    including references to the builder, kernel metadata, and mappings
    between FX values and MLIR SSA values.
    
    Most kernel metadata is accessed via property methods that derive
    values directly from bound_kernel, avoiding redundant storage.
    """
    
    def __init__(self, bound_kernel: Any):
        """Initialize a LoweringContext from a bound Helion kernel.
        
        Args:
            bound_kernel: The bound Helion kernel
        """
        import torch
        
        self.bound_kernel = bound_kernel
        self.builder = MLIRBuilder()
        
        # Derive kernel name from the kernel function
        self.kernel_name = bound_kernel.kernel.fn.__name__
        
        # Extract kernel arguments upfront
        self._kernel_args_cache = _extract_kernel_args(bound_kernel)
        tensor_args = [arg for arg in self._kernel_args_cache if arg.is_tensor]
        
        if not tensor_args:
            raise ValueError("No tensor arguments found")
        
        # Update kernel_args with actual shapes and types from fake_args
        fake_args = bound_kernel.fake_args
        for arg in self._kernel_args_cache:
            if arg.is_tensor and arg.index < len(fake_args):
                tensor = fake_args[arg.index]
                # Use actual tensor shape
                arg.shape = [int(s) if not isinstance(s, torch.SymInt) else None 
                             for s in tensor.shape]
                # Derive element type from THIS tensor's dtype
                element_type = torch_dtype_to_mlir_element_type(tensor.dtype)
                arg.mlir_type = format_tensor_type(arg.shape, element_type)
        
        # Build loop information from block sizes
        self.loop_extents: dict[str, int] = {}
        
        # Pre-compute total extents for all blocks
        fake_args = bound_kernel.fake_args
        tensor_args = self.get_tensor_args()
        
        for info in bound_kernel.env.block_sizes:
            block_name = first_debug_name(info.debug_names, fallback=f"block_{info.block_id}")
            total_extent = _resolve_extent(bound_kernel, info, fake_args, tensor_args)
            self.loop_extents[block_name] = total_extent
        
        # Mutable state that gets populated during lowering
        self.symbols: dict[str, str] = {}
        self.host_tensors: dict[str, str] = {}
        
        # SSA Value Tracking (populated during graph walking by IRVisitor)
        self.node_values: dict[str, str] = {}      # FX node name → MLIR SSA value
        self.node_types: dict[str, str] = {}       # FX node name → MLIR type string
        self.initial_acc_ssa: dict[str, str] = {}  # Accumulator node → initial SSA (for phi)
        
        # Graph Registry (populated before walking starts)
        self.graphs: dict[int, Any] = {}           # Graph ID → ForLoopGraphInfo
        
        # Pre-computed MLIR values (computed before graph walking)
        self.block_size_ssa: dict[int, str] = {}   # Block ID → block size SSA value
        self.reduction_trip_counts: dict[int, str] = {}  # Block ID → trip count SSA
        
        # Multi-value loop tracking
        self.loop_result_values: dict[str, Any] = {}  # Loop name → result info
    
    # -------------------------------------------------------------------------
    # Property methods - derive values from bound_kernel
    # -------------------------------------------------------------------------
    
    @property
    def device_ir(self) -> Any:
        """Get DeviceIR from bound_kernel."""
        return self.bound_kernel.host_function.device_ir
    
    @property
    def block_sizes(self) -> dict[int, Any]:
        """Get block sizes as dict from bound_kernel.env."""
        return {info.block_id: info for info in self.bound_kernel.env.block_sizes}
    
    @property
    def parallel_block_ids(self) -> list[int]:
        """Get grid block IDs for parallel loops."""
        grid_block_groups = self.device_ir.grid_block_ids
        if not grid_block_groups:
            return []
        return list(grid_block_groups[0])
    
    @property
    def env(self) -> Any:
        """Get CompileEnvironment from bound_kernel."""
        return self.bound_kernel.env
    
    @property
    def kernel_args(self) -> list[KernelArgInfo]:
        """Get kernel argument info (pre-computed during __init__)."""
        return self._kernel_args_cache
    
    def get_tensor_args(self) -> list[KernelArgInfo]:
        """Get only the tensor arguments from kernel_args."""
        return [arg for arg in self.kernel_args if arg.is_tensor]

    
    def get_module_attributes(self) -> dict[str, tuple[object, str]]:
        """Get block sizes as module attributes.
        
        Returns a dict mapping attribute name to (value, type) for module attributes.
        Uses -1 for undefined (symbolic) block sizes.
        
        Naming convention:
        - loom.block_size_0, loom.block_size_1, ... = tile sizes for each block ID
        """
        attrs = {}
        # Emit block sizes using block_id-based naming
        for info in self.env.block_sizes:
            attr_name = f"loom.block_size_{info.block_id}"
            
            # Use size directly if concrete int
            if isinstance(info.size, int):
                attrs[attr_name] = (info.size, "index")
            else:
                # Use -1 for undefined/symbolic sizes
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


def _resolve_extent(
    bound_kernel: Any,
    info: Any,
    fake_args: tuple,
    tensor_args: list[KernelArgInfo],
) -> int:
    """Resolve extent for a named dimension.
    
    Extracts the dimension extent from BlockSizeInfo.size, which contains
    the symbolic or concrete size of the dimension associated with this block.
    This is set when hl.tile(tensor.size(dim)) is called.
    
    Args:
        bound_kernel: The bound kernel for environment access
        info: BlockSizeInfo containing dimension size information
        fake_args: Fake tensor arguments (fallback)
        tensor_args: KernelArgInfo list (fallback)
    
    Returns:
        The concrete integer extent for this dimension
    """
    import torch
    
    # Primary strategy: use BlockSizeInfo.size directly
    # info.size contains the dimension's total size (e.g., from x.size(0))
    if hasattr(info, 'size') and info.size is not None:
        size = info.size
        if isinstance(size, int):
            return size
        elif isinstance(size, torch.SymInt):
            # Get the concrete hint value from the CompileEnvironment
            env = bound_kernel.env
            return env.size_hint(size)
    
    # Fallback: use block_id to look up via CompileEnvironment.get_block_id
    # This resolves the block ID from dimension symbols in tensor shapes
    if hasattr(info, 'block_id'):
        block_id = info.block_id
        env = bound_kernel.env
        # Look through tensor arguments to find matching dimension
        for arg in tensor_args:
            if arg.index < len(fake_args):
                tensor = fake_args[arg.index]
                for dim in range(tensor.ndim):
                    dim_size = tensor.size(dim)
                    resolved_block_id = env.get_block_id(dim_size)
                    if resolved_block_id == block_id:
                        return env.size_hint(dim_size)
    
    # Ultimate fallback: return 1 as a safe default
    return 1


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
