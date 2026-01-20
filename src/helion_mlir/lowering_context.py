"""Lowering context for FX-to-MLIR conversion.

This module defines the LoweringContext dataclass that holds all state
needed during the lowering process, including the MLIR builder, kernel
metadata, and mappings between FX nodes and MLIR SSA values.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Sequence

from .mlir_utils import (
    MLIROutputHelper,
    torch_dtype_to_mlir_element_type,
    format_tensor_type,
)

if TYPE_CHECKING:
    from torch import Tensor
    import torch





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
        self.builder = MLIROutputHelper()
        # Always output linalg-on-tensors for now as per user request
        
        # Derive kernel name from the kernel function
        self.kernel_name = bound_kernel.kernel.fn.__name__
        
        # Get parameter names from the kernel signature
        param_names = list(bound_kernel.kernel.signature.parameters.keys())
        fake_args = bound_kernel.fake_args
        
        # Build arg_mlir_types: maps arg_name -> MLIR type string (for tensor args only)
        # BoundKernel.fake_args types:
        #   - FakeTensor: regular tensor argument -> becomes MLIR function parameter
        #   - int/float: ConstExpr compile-time constant -> not in function signature
        self.arg_mlir_types: dict[str, str] = {}
        
        for name, fake_arg in zip(param_names, fake_args):
            if isinstance(fake_arg, torch.Tensor):
                shape = [
                    int(s) if not isinstance(s, torch.SymInt) else None
                    for s in fake_arg.shape
                ]
                element_type = torch_dtype_to_mlir_element_type(fake_arg.dtype)
                self.arg_mlir_types[name] = format_tensor_type(shape, element_type)
        
        if not self.arg_mlir_types:
            raise ValueError("No tensor arguments found")
        

        
        # Build loop extents from block sizes
        # Key: block_id, Value: concrete extent (from shape_env)
        self.loop_extents: dict[int, int] = {}
        shape_env = bound_kernel.env.shape_env
        
        for info in bound_kernel.env.block_sizes:
            size = info.size
            if isinstance(size, int):
                self.loop_extents[info.block_id] = size
            elif isinstance(size, torch.SymInt):
                # Look up concrete value in shape_env
                sym = size._sympy_()
                if sym in shape_env.var_to_val:
                    self.loop_extents[info.block_id] = int(shape_env.var_to_val[sym])
                else:
                    raise RuntimeError(f"Cannot resolve extent for block {info.block_id}")
            else:
                raise RuntimeError(f"Unsupported size type: {type(size)} for block {info.block_id}")
        
        # Mutable state that gets populated during lowering
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
        
        # Pre-computed host tensor types (computed by scanning all graphs)
        # Maps tensor name -> MLIR type string (e.g., "tensor<128x128xf32>")
        self.host_tensor_types: dict[str, str] = {}
        self._precompute_host_tensor_types()
    
    def _precompute_host_tensor_types(self) -> None:
        """Pre-compute MLIR types for all _host_tensor nodes by scanning all graphs.
        
        Uses origin-based logic to determine dynamic vs concrete dimensions:
        - BlockSizeOrigin -> dynamic '?'
        - Other with concrete value in shape_env -> concrete int
        - Unknown -> dynamic '?'
        """
        import helion.language._tracing_ops as hl_tracing_ops
        
        device_ir = self.bound_kernel.host_function.device_ir
        
        # Find rolled reduction graph IDs to skip
        rolled_ids = {
            info.new_graph_id 
            for info in device_ir.rolled_reductions 
            if info.new_graph_id is not None
        }
        
        seen: set[str] = set()
        
        for graph_info in device_ir.graphs:
            if graph_info.graph_id in rolled_ids:
                continue
            for node in graph_info.graph.nodes:
                if node.op == "call_function" and node.target is hl_tracing_ops._host_tensor:
                    name = node.args[0]
                    if name not in seen:
                        seen.add(name)
                        
                        # Get FakeTensor from node metadata
                        fake_tensor = node.meta.get("val")
                        mlir_type = self.compute_mlir_type_from_fake_tensor(fake_tensor)
                        self.host_tensor_types[name] = mlir_type
    
    def compute_mlir_type_from_fake_tensor(self, fake_tensor, dtype: str = "f32") -> str:
        """Compute MLIR tensor type from a FakeTensor using origin-based logic.
        
        For each dimension:
        - BlockSizeOrigin -> dynamic '?'
        - Other with concrete value in shape_env -> concrete int
        - Unknown -> dynamic '?'
        
        Args:
            fake_tensor: A FakeTensor from node.meta['val']
            dtype: Element type string (default "f32")
            
        Returns:
            MLIR tensor type string like "tensor<128x256xf32>" or "tensor<?x?xf32>"
        """
        from helion._compiler.variable_origin import BlockSizeOrigin
        
        if fake_tensor is None or not hasattr(fake_tensor, "shape"):
            raise RuntimeError("FakeTensor is None or does not have shape")
        
        host_function = self.bound_kernel.host_function
        shape_env = self.bound_kernel.env.shape_env
        
        # Determine shape: ? for BlockSizeOrigin, concrete otherwise
        shape = []
        ndim = len(fake_tensor.shape)
        for dim_idx in range(ndim):
            dim_size = fake_tensor.shape[dim_idx]
            if hasattr(dim_size, '_sympy_'):
                # This is a SymInt - check its origin
                sym = dim_size._sympy_()
                origin_info = host_function.expr_to_origin.get(sym)
                origin = origin_info.origin if origin_info else None
                
                if isinstance(origin, BlockSizeOrigin):
                    # Dynamic dimension
                    shape.append(None)
                elif sym in shape_env.var_to_val:
                    # Concrete value from shape_env
                    shape.append(int(shape_env.var_to_val[sym]))
                else:
                    # Unknown - use dynamic
                    shape.append(None)
            elif isinstance(dim_size, int):
                # Already concrete
                shape.append(int(dim_size))
            else:
                # Unknown type - use dynamic
                shape.append(None)
        
        return format_tensor_type(shape, dtype)
    
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
