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



class LoweringContext:
    """Context passed to lowering functions during FX-to-MLIR conversion.
    
    This class holds all the state needed during the lowering process,
    including references to the builder, kernel metadata, and mappings
    between FX values and MLIR SSA values.
    
    Most kernel metadata is accessed via property methods that derive
    values directly from bound_kernel, avoiding redundant storage.
    """
    
    def __init__(
        self,
        builder: MLIRBuilder,
        bound_kernel: Any,
    ):
        """Initialize a LoweringContext.
        
        Args:
            builder: MLIRBuilder for text emission
            bound_kernel: The bound Helion kernel
        """
        self.builder = builder
        self.bound_kernel = bound_kernel
        
        # Mutable state that gets populated during lowering
        self.loop_extents: dict[str, int] = {}
        self.grid_loops: list[LoopInfo] = []
        self.inner_loops: list[LoopInfo] = []
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
        
        # Cached computed values (lazily populated)
        self._kernel_args_cache: list[KernelArgInfo] | None = None
    
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
        """Get kernel argument info, with caching.
        
        This is cached because it requires computation and may be accessed
        multiple times during lowering.
        """
        if self._kernel_args_cache is None:
            self._kernel_args_cache = _extract_kernel_args(self.bound_kernel)
        return self._kernel_args_cache
    
    # -------------------------------------------------------------------------
    # Factory method
    # -------------------------------------------------------------------------
    
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
        
        # Extract kernel args
        kernel_args = _extract_kernel_args(bound_kernel)
        tensor_args = [arg for arg in kernel_args if arg.is_tensor]
        
        if not tensor_args:
            raise ValueError("No tensor arguments found")
        
        # Create context
        ctx = cls(
            builder=builder,
            bound_kernel=bound_kernel,
        )
        
        # Pre-populate kernel_args cache since we already computed it
        ctx._kernel_args_cache = kernel_args
        
        # Build loop information - this populates loop_extents
        ctx._build_loop_info()
        
        # Update kernel_args with actual shapes and types from fake_args
        for arg in ctx.kernel_args:
            if arg.is_tensor and arg.index < len(fake_args):
                tensor = fake_args[arg.index]
                # Use actual tensor shape
                arg.shape = [int(s) if not isinstance(s, torch.SymInt) else None 
                             for s in tensor.shape]
                # Derive element type from THIS tensor's dtype
                element_type = torch_dtype_to_mlir_element_type(tensor.dtype)
                arg.mlir_type = format_tensor_type(arg.shape, element_type)
        
        return ctx
    
    def get_tensor_args(self) -> list[KernelArgInfo]:
        """Get only the tensor arguments from kernel_args."""
        return [arg for arg in self.kernel_args if arg.is_tensor]

    
    def _build_loop_info(self) -> None:
        """Build loop information from block sizes 
        
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
            
            self.grid_loops.append(LoopInfo(
                block_id=block_id,
                name=block_name,
                tile_size=tile_size,
                trip_count=trip_count,
                total_extent=total_extent,
                is_symbolic=is_symbolic,
                iv_name=f"%iv_{block_name}"
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
            
            self.inner_loops.append(LoopInfo(
                block_id=block_id,
                name=block_name,
                tile_size=tile_size,
                trip_count=trip_count,
                total_extent=total_extent,
                is_symbolic=is_symbolic,
                iv_name=f"%iv_{block_name}"
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
        """Resolve extent for a named dimension.
        
        Extracts the dimension extent from BlockSizeInfo.size, which contains
        the symbolic or concrete size of the dimension associated with this block.
        This is set when hl.tile(tensor.size(dim)) is called.
        
        Args:
            block_name: Debug name for the block (unused, kept for compatibility)
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
                env = self.bound_kernel.env
                return env.size_hint(size)
        
        # Fallback: use block_id to look up via CompileEnvironment.get_block_id
        # This resolves the block ID from dimension symbols in tensor shapes
        if hasattr(info, 'block_id'):
            block_id = info.block_id
            env = self.bound_kernel.env
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
    
    def get_module_attributes(self) -> dict[str, tuple[object, str]]:
        """Get block sizes as module attributes.
        
        Returns a dict mapping attribute name to (value, type) for module attributes.
        Uses -1 for undefined (symbolic) block sizes.
        
        Naming convention:
        - loom.block_size_0, loom.block_size_1, ... = tile sizes for each block ID
        """
        attrs = {}
        # Consolidate loop vars by iterating order
        all_loops = self.grid_loops + self.inner_loops
        # Emit block sizes using block_id-based naming
        for loop in all_loops:
            if loop.block_id is None: # Skip loops without a block_id (e.g., grid loops from outer_loop_names)
                continue
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
