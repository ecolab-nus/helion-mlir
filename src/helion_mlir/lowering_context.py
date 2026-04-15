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
    format_memref_type,
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
        self.mlir_output_helper = MLIROutputHelper()
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
        for info in bound_kernel.env.block_sizes:
            extent = self._resolve_block_extent(info)
            if extent is not None:
                self.loop_extents[info.block_id] = extent
        
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

    def _resolve_block_extent(self, info: Any) -> int | None:
        """Resolve concrete extent for one block if possible."""
        shape_env = self.bound_kernel.env.shape_env
        size = info.size

        if isinstance(size, int):
            return int(size)

        if hasattr(size, "_sympy_"):
            size_sym = size._sympy_()
            if size_sym in shape_env.var_to_val:
                return int(shape_env.var_to_val[size_sym])

        var = getattr(info, "var", None)
        if var is not None and hasattr(var, "_sympy_"):
            var_sym = var._sympy_()
            if var_sym in shape_env.var_to_val:
                return int(shape_env.var_to_val[var_sym])

        return None

    def get_loop_extent(self, block_id: int) -> int | None:
        """Return resolved block extent, if available."""
        return self.loop_extents.get(block_id)

    def get_loop_extent_or_default(self, block_id: int, default: int = 1) -> int:
        """Return the resolved block extent, or the provided default."""
        extent = self.get_loop_extent(block_id)
        if extent is not None:
            return extent

        return default
    
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
                        mlir_type = self.compute_mlir_memref_type_from_fake_tensor(fake_tensor)
                        self.host_tensor_types[name] = mlir_type
    
    def compute_mlir_type_from_fake_tensor(self, fake_tensor) -> str:
        """Compute MLIR tensor type from a FakeTensor using origin-based logic.
        
        For each dimension:
        - BlockSizeOrigin -> dynamic '?'
        - Other with concrete value in shape_env -> concrete int
        - Unknown -> dynamic '?'
        
        Args:
            fake_tensor: A FakeTensor from node.meta['val']
            
        Returns:
            MLIR tensor type string like "tensor<128x256xf32>" or "tensor<?x?xf32>"
        """
        from helion._compiler.variable_origin import BlockSizeOrigin
        from .mlir_utils import torch_dtype_to_mlir_element_type
        
        if fake_tensor is None or not hasattr(fake_tensor, "shape"):
            raise RuntimeError("FakeTensor is None or does not have shape")
            
        dtype_str = torch_dtype_to_mlir_element_type(fake_tensor.dtype)
        
        host_function = self.bound_kernel.host_function
        shape_env = self.bound_kernel.env.shape_env
        
        # Determine shape: ? for dynamic BlockSizeOrigin, concrete otherwise
        shape = []
        block_sizes = self.bound_kernel.env.block_sizes
        ndim = len(fake_tensor.shape)
        for dim_idx in range(ndim):
            dim_size = fake_tensor.shape[dim_idx]
            if hasattr(dim_size, '_sympy_'):
                # This is a SymInt - check its origin
                sym = dim_size._sympy_()
                origin_info = host_function.expr_to_origin.get(sym)
                origin = origin_info.origin if origin_info else None

                if isinstance(origin, BlockSizeOrigin):
                    # Only mark as dynamic if the block size is not a static integer.
                    # hl.specialize and reduction-loop blocks have a concrete int size
                    # (e.g. head_dim=128, num_heads=32); those should be emitted as
                    # static dimensions, matching the behaviour of resolve_dimension().
                    block_info = block_sizes[origin.block_id]
                    if isinstance(block_info.size, int):
                        shape.append(block_info.size)
                    else:
                        shape.append(None)
                elif sym.is_number:
                    # The sympy expression reduced to a concrete integer constant
                    # (e.g. hl.specialize resolves head_dim to Integer(128)).
                    # shape_env.var_to_val only holds symbolic variables so this
                    # case must be caught explicitly before the var_to_val lookup.
                    shape.append(int(sym))
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

        return format_tensor_type(shape, dtype_str)
    
    def compute_mlir_memref_type_from_fake_tensor(self, fake_tensor) -> str:
        """Compute MLIR memref type from a FakeTensor using origin-based logic.
        
        For each dimension:
        - BlockSizeOrigin -> dynamic '?'
        - Other with concrete value in shape_env -> concrete int
        - Unknown -> dynamic '?'
        
        Args:
            fake_tensor: A FakeTensor from node.meta['val']
            
        Returns:
            MLIR memref type string like "memref<128x256xf32>" or "memref<?x?xf32>"
        """
        from helion._compiler.variable_origin import BlockSizeOrigin
        from .mlir_utils import torch_dtype_to_mlir_element_type
        
        if fake_tensor is None or not hasattr(fake_tensor, "shape"):
            raise RuntimeError("FakeTensor is None or does not have shape")
            
        dtype_str = torch_dtype_to_mlir_element_type(fake_tensor.dtype)
        
        host_function = self.bound_kernel.host_function
        shape_env = self.bound_kernel.env.shape_env
        
        # Determine shape: ? for dynamic BlockSizeOrigin, concrete otherwise
        shape = []
        block_sizes = self.bound_kernel.env.block_sizes
        ndim = len(fake_tensor.shape)
        for dim_idx in range(ndim):
            dim_size = fake_tensor.shape[dim_idx]
            if hasattr(dim_size, '_sympy_'):
                # This is a SymInt - check its origin
                sym = dim_size._sympy_()
                origin_info = host_function.expr_to_origin.get(sym)
                origin = origin_info.origin if origin_info else None

                if isinstance(origin, BlockSizeOrigin):
                    # Only mark as dynamic if the block size is not a static integer.
                    # hl.specialize and reduction-loop blocks have a concrete int size;
                    # those should appear as static dimensions in the memref type.
                    block_info = block_sizes[origin.block_id]
                    if isinstance(block_info.size, int):
                        shape.append(block_info.size)
                    else:
                        shape.append(None)
                elif sym.is_number:
                    # The sympy expression reduced to a concrete integer constant.
                    # Must be caught before the var_to_val lookup since concrete
                    # integers are not stored as symbolic variables.
                    shape.append(int(sym))
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
        
        return format_memref_type(shape, dtype_str)
    
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
        """Get grid block IDs for the first parallel loop (backward compat)."""
        grid_block_groups = self.device_ir.grid_block_ids
        if not grid_block_groups:
            return []
        return list(grid_block_groups[0])

    @property
    def all_grid_block_ids(self) -> list[list[int]]:
        """Get block IDs for every grid group."""
        return [list(g) for g in self.device_ir.grid_block_ids]

    @property
    def env(self) -> Any:
        """Get CompileEnvironment from bound_kernel."""
        return self.bound_kernel.env

    def _build_block_id_alias_map(self) -> dict[int, int]:
        """Detect block IDs across grid groups that tile the same source dimension.

        Two blocks are considered aliases when they share the same
        ``debug_names`` **and** the same ``numel`` (total extent).  The first
        block seen (in grid-group order) becomes the *canonical* representative;
        later blocks map to that canonical ID.

        Returns:
            Mapping from every block_id to its canonical block_id.
        """
        block_info_map = {info.block_id: info for info in self.env.block_sizes}
        canonical: dict[tuple[frozenset[str], int], int] = {}
        alias: dict[int, int] = {}

        for grid_group in self.all_grid_block_ids:
            for bid in grid_group:
                info = block_info_map.get(bid)
                if info is None:
                    alias[bid] = bid
                    continue
                key = (frozenset(info.debug_names), info.numel)
                if key not in canonical:
                    canonical[key] = bid
                alias[bid] = canonical[key]

        for info in self.env.block_sizes:
            if info.block_id not in alias:
                alias[info.block_id] = info.block_id

        return alias

    @property
    def block_id_alias(self) -> dict[int, int]:
        """Lazily computed block-ID alias map (cached)."""
        if not hasattr(self, "_block_id_alias"):
            self._block_id_alias = self._build_block_id_alias_map()
        return self._block_id_alias

    def resolve_block_id(self, block_id: int) -> int:
        """Resolve a block_id to its canonical (first-seen) equivalent."""
        return self.block_id_alias.get(block_id, block_id)

    def get_module_attributes(
        self,
        *,
        used_canonical_ids: set[int] | None = None,
    ) -> dict[str, tuple[object, str]]:
        """Get block sizes as module attributes.

        Only emits attributes for *canonical* blocks with symbolic sizes.
        Aliased blocks share the same attribute as their canonical block.
        """
        alias = self.block_id_alias
        seen_canonical: set[int] = set()
        attrs: dict[str, tuple[object, str]] = {}

        for info in self.env.block_sizes:
            if isinstance(info.size, int):
                continue
            canonical_id = alias.get(info.block_id, info.block_id)
            if (
                used_canonical_ids is not None
                and canonical_id not in used_canonical_ids
            ):
                continue
            if canonical_id in seen_canonical:
                continue
            seen_canonical.add(canonical_id)

            sym_name = next(iter(info.debug_names), f"block_{canonical_id}")
            upper_bound = self.get_loop_extent(info.block_id)
            if upper_bound is None:
                continue
            is_reduction = str(info.reduction).lower()
            attr_name = f"loom.{sym_name}"
            dict_val = (
                f'{{upper_bound = {upper_bound} : index, '
                f'is_reduction = {is_reduction}}}'
            )
            attrs[attr_name] = (dict_val, "")

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
