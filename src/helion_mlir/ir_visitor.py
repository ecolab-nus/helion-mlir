"""IR Visitor for walking Device IR graphs and generating MLIR.

This module implements a visitor pattern for converting Helion Device IR
to MLIR by walking FX graph nodes instruction-by-instruction.

The visitor dispatches to specific handlers based on the node's target:
- _get_symnode -> SSA lookup via Origin (BlockSizeOrigin -> block_size_ssa)
- full -> tensor.empty + linalg.fill
- _for_loop -> scf.for + recursive visit
- _phi -> Loop result SSA (simplified merge pattern detection)
- _host_tensor -> function argument mapping
- aten.sym_size.int -> inline concrete value or tensor.dim
- load -> memref.subview + bufferization.to_tensor
- store -> bufferization.to_memref + memref.copy
- atomic_add -> memref.subview + loom.sum
- aten.* compute -> linalg-on-tensors (via torch-mlir)
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any

import torch
import torch.fx as fx
from torch.ops import aten

import helion.language.memory_ops as hl_memory_ops
import helion.language.atomic_ops as hl_atomic_ops
import helion.language._tracing_ops as hl_tracing_ops
import helion.language.creation_ops as hl_creation_ops
import helion.language.view_ops as hl_view_ops
import helion.language.tile_ops as hl_tile_ops

from .handlers import (
    register_compute_handlers,
    register_control_flow_handlers,
    register_memory_handlers,
    register_symbol_handlers,
    register_tensor_handlers,
    register_tile_handlers,
)
_custome_ops_cache: dict | None = None


def _get_custome_ops() -> dict:
    """Lazily load custom ops from custome_op package (optional).

    Returns a dict mapping handler name -> function object.
    Returns an empty dict if the package is not available.

    Note: ``custome_op/__init__.py`` re-exports ops from submodules, which can
    shadow module names in the package namespace. We use
    ``importlib.import_module`` to access real submodules and retrieve the
    decorated function objects.
    """
    global _custome_ops_cache
    if _custome_ops_cache is not None:
        return _custome_ops_cache

    import importlib, pathlib, sys
    custome_op_dir = str(pathlib.Path(__file__).resolve().parents[2])  # helion-mlir root
    if custome_op_dir not in sys.path:
        sys.path.insert(0, custome_op_dir)
    try:
        gather_mod = importlib.import_module("custome_op.gather")
        broadcast_mod = importlib.import_module("custome_op.broadcast")
        _custome_ops_cache = {
            "gather": gather_mod.gather,
            "broadcast": broadcast_mod.broadcast,
        }
    except Exception:
        _custome_ops_cache = {}
    return _custome_ops_cache

from .mlir_utils import (
    format_attr_dict,
    format_string_attr,
    torch_dtype_to_mlir_element_type,
    format_memref_type,
)
from .torch_mlir_helper import (
    TorchMLIRNodeImporter,
    import_aten_node_to_mlir,
)

if TYPE_CHECKING:
    from helion._compiler.device_ir import ForLoopGraphInfo, RootGraphInfo
    from .lowering_context import LoweringContext


class IRVisitor:
    """Visits FX graph nodes and generates MLIR operations.
    
    This visitor walks the Device IR instruction-by-instruction, generating
    corresponding MLIR operations. It maintains a mapping from FX node names
    to MLIR SSA values.
    
    Usage:
        visitor = IRVisitor(ctx)
        visitor.register_graph(0, for_loop_graph)
        visitor.visit_graph(root_graph)
    """
    
    def __init__(self, ctx: "LoweringContext"):
        self.ctx = ctx
        self.mlir_output_helper = ctx.mlir_output_helper
        
        # Loop-local state (managed per loop, not persisted globally)
        # These are reset/managed during visit_for_loop calls
        self.loop_iter_args: dict[str, str] = {}  # placeholder name → SSA (per loop context)
        self.current_loop_result: str | list[str] | None = None  # Set by inner graph output
        self.loop_depth: int = 0  # Depth tracking for nested loops
        self.current_block_id: int | None = None  # Current loop's block_id for IV reference
        # Nodes representing contiguous 1-D index ranges (tile_index-based patterns)
        self.range_index_block_ids: dict[str, int] = {}
        # Active scf.for bounds keyed by canonical block_id.
        self.loop_bounds: dict[int, tuple[str, str]] = {}
        self._direct_handlers: dict[object, str] = {}
        self._predicate_handlers: list[tuple[object, str]] = []
        register_symbol_handlers(self._direct_handlers, self._predicate_handlers)
        register_tile_handlers(self._direct_handlers, self._predicate_handlers)
        register_control_flow_handlers(self._direct_handlers, self._predicate_handlers)
        register_memory_handlers(self._direct_handlers, self._predicate_handlers)
        register_tensor_handlers(self._direct_handlers, self._predicate_handlers)
        register_compute_handlers(self._direct_handlers, self._predicate_handlers)
    
    def resolve_dimension(self, dim_size, dim_hint: int = 0, overrides: dict[int, str] | None = None) -> tuple[str, bool]:
        """Resolve a dimension size to an SSA value or inline literal.

        Block IDs are resolved through the alias map so that dimensions
        coming from the same source across grids share the same SSA.
        
        Args:
            dim_size: The dimension size to resolve (int or sympy expression)
            dim_hint: The dimension index (used for overrides)
            overrides: Optional mapping of dim_index -> SSA string
        """
        return self.ctx.symbol_resolver.resolve_dimension(
            dim_size,
            dim_hint,
            overrides=overrides,
        )

    def _propagate_metadata(self, src_name: str, dst_name: str, ssa_override: str | None = None, is_placeholder: bool = False):
        """Propagate metadata (values, types, and overrides) from one node to another.
        
        This is used when mapping outer-scope variables to inner-scope placeholders
        in conditional blocks or loop bodies, and for passthrough operators.
        """
        if ssa_override:
            ssa = ssa_override
        else:
            ssa = self.ctx.node_values.get(src_name, f"%{src_name}")

        if is_placeholder:
            self.loop_iter_args[dst_name] = ssa

        self.ctx.bind_node_value(dst_name, ssa)

        if src_name in self.ctx.node_types:
            self.ctx.node_types[dst_name] = self.ctx.node_types[src_name]
            
        if src_name in self.ctx.gather_dim_overrides:
            self.ctx.gather_dim_overrides[dst_name] = self.ctx.gather_dim_overrides[src_name]

    @staticmethod
    def _split_top_level_commas(text: str) -> list[str]:
        """Split a type payload on commas that are not nested inside delimiters."""
        parts: list[str] = []
        current: list[str] = []
        angle_depth = 0
        square_depth = 0
        paren_depth = 0

        for ch in text:
            if ch == "<":
                angle_depth += 1
            elif ch == ">":
                angle_depth = max(0, angle_depth - 1)
            elif ch == "[":
                square_depth += 1
            elif ch == "]":
                square_depth = max(0, square_depth - 1)
            elif ch == "(":
                paren_depth += 1
            elif ch == ")":
                paren_depth = max(0, paren_depth - 1)

            if ch == "," and angle_depth == 0 and square_depth == 0 and paren_depth == 0:
                parts.append("".join(current).strip())
                current = []
                continue

            current.append(ch)

        if current:
            parts.append("".join(current).strip())
        return parts

    @classmethod
    def _parse_plain_memref_dimensions(cls, memref_type: str, op_name: str) -> list[str]:
        """Extract dimensions from a memref type supported by current subview lowering."""
        from .resolvers import TypeResolver

        dummy_resolver = TypeResolver.__new__(TypeResolver)
        return TypeResolver.parse_plain_memref_dimensions(dummy_resolver, memref_type, op_name)

    @staticmethod
    def _validate_full_unit_slice(idx: slice, op_name: str, dim: int) -> None:
        """Reject slices that the current lowering does not model correctly."""
        if idx.start is not None or idx.stop is not None or idx.step not in (None, 1):
            raise RuntimeError(
                f"{op_name} only supports full-dimension unit-step slices (:). "
                f"Unsupported slice at dim {dim}: slice({idx.start}, {idx.stop}, {idx.step})"
            )
        
    # def register_graph(self, graph_id: int, graph_info: "GraphInfo") -> None:
    #     """Register a graph for later visitation (e.g., ForLoopGraphInfo)."""
    #     self.ctx.graphs[graph_id] = graph_info
    
    def visit_graph(self, graph_info: "GraphInfo") -> None:
        """Visit all nodes in a graph in order."""
        for node in graph_info.graph.nodes:
            self.visit_node(node)
    
    def visit_node(self, node: fx.Node) -> str | None:
        """Dispatch to appropriate handler based on node type and target."""
        if node.op == "placeholder":
            return self.visit_placeholder(node)
        elif node.op == "call_function":
            return self.visit_call_function(node)
        elif node.op == "output":
            return self.visit_output(node)
        elif node.op == "get_attr":
            return self.visit_get_attr(node)
        else:
            # Unknown op, skip or warn
            return None
    
    def visit_placeholder(self, node: fx.Node) -> str:
        """Handle placeholder nodes (function arguments or loop iter_args)."""
        # Check if this is a loop iter_arg
        if node.name in self.loop_iter_args:
            ssa = self.loop_iter_args[node.name]
            self.ctx.bind_node_value(node.name, ssa)
            return ssa
        
        # Otherwise it's a function argument placeholder
        # For now, just create a placeholder SSA value
        ssa = f"%{node.name}"
        self.ctx.bind_node_value(node.name, ssa)
        
        # Register type from arg_mlir_types if available
        if node.name in self.ctx.arg_mlir_types:
            self.ctx.node_types[node.name] = self.ctx.arg_mlir_types[node.name]
                
        return ssa
    
    def visit_call_function(self, node: fx.Node) -> str | None:
        """Dispatch call_function nodes to specific handlers."""
        target = node.target

        method_name = self._direct_handlers.get(target)
        if method_name is not None:
            return getattr(self, method_name)(node)

        if self._is_python_index_arith_target(target):
            return self.visit_python_index_arith(node)

        _custom = _get_custome_ops()
        if target is _custom.get("gather"):
            return self.visit_gather(node)
        if target is _custom.get("broadcast"):
            return self.visit_broadcast(node)

        if target is aten.add.Tensor:
            maybe_index_base = self.visit_tile_index_add(node)
            if maybe_index_base is not None:
                return maybe_index_base

        if target is getattr(__builtins__, 'getitem', None) or \
           (hasattr(target, '__name__') and target.__name__ == 'getitem'):
            return self.visit_getitem(node)

        import operator as _operator
        _COMPARISON_OPS = {
            _operator.eq: "eq",
            _operator.ne: "ne",
            _operator.lt: "slt",
            _operator.le: "sle",
            _operator.gt: "sgt",
            _operator.ge: "sge",
        }
        if target in _COMPARISON_OPS:
            return self.visit_comparison(node, _COMPARISON_OPS[target])

        for predicate, handler_name in self._predicate_handlers:
            if predicate(target):
                return getattr(self, handler_name)(node)

        raise RuntimeError(f"Unsupported target: {target}")

    
    def visit_output(self, node: fx.Node) -> str | None:
        """Handle output nodes."""
        args = node.args[0] if node.args else None
        
        if args is None:
            return None
        
        # For ForLoopGraphInfo, the output is the yield value(s)
        if isinstance(args, (list, tuple)):
            if len(args) == 1:
                arg = args[0]
                if isinstance(arg, fx.Node):
                    result = self.ctx.node_values.get(arg.name)
                    self.current_loop_result = [result]  # Store as list for consistency
                    return result
            # Multiple outputs - store ALL results
            results = []
            for arg in args:
                if isinstance(arg, fx.Node):
                    results.append(self.ctx.node_values.get(arg.name))
            self.current_loop_result = results  # Store entire list
            return results[0] if results else None

        
        if isinstance(args, fx.Node):
            self.current_loop_result = self.ctx.node_values.get(args.name)
            return self.current_loop_result
        
        return None
    
    def visit_get_attr(self, node: fx.Node) -> str:
        """Handle get_attr nodes."""
        # Just create a placeholder SSA
        ssa = self.mlir_output_helper.fresh(node.name)
        self.ctx.bind_node_value(node.name, ssa)
        return ssa
    
    # -------------------------------------------------------------------------
    # Specific Node Handlers
    # -------------------------------------------------------------------------
    
    def _try_get_block_id_from_node(self, node: fx.Node) -> int | None:
        """Try to extract the *canonical* block_id from a node's BlockSizeOrigin.

        The returned ID is resolved through the alias map so that blocks
        from different grids that tile the same source dimension share the
        same IV / block-size SSA.
        """
        return self.ctx.symbol_resolver.try_get_block_id_from_node(
            node,
            self.range_index_block_ids,
        )

    def _get_block_size_value(self, canonical_block_id: int) -> tuple[str, bool]:
        """Return block-size value as (value, is_static_literal)."""
        return self.ctx.symbol_resolver.get_block_size_value(canonical_block_id)

    def _is_singleton_block(self, canonical_block_id: int) -> bool:
        """Whether block size is known to be exactly 1."""
        return self.ctx.symbol_resolver.is_singleton_block(canonical_block_id)

    def _extract_tile_index_add(self, node: fx.Node) -> tuple[int, Any] | None:
        """Detect aten.add(tile_index(...), base) pattern used for contiguous ranges."""
        if node.target is not aten.add.Tensor or len(node.args) < 2:
            return None
        lhs, rhs = node.args[0], node.args[1]

        if isinstance(lhs, fx.Node) and lhs.target is hl_tile_ops.tile_index:
            block_id = self._try_get_block_id_from_node(lhs)
            if block_id is not None:
                return (block_id, rhs)
        if isinstance(rhs, fx.Node) and rhs.target is hl_tile_ops.tile_index:
            block_id = self._try_get_block_id_from_node(rhs)
            if block_id is not None:
                return (block_id, lhs)
        return None
    
    def visit_get_symnode(self, node: fx.Node) -> str:
        """Look up pre-emitted SSA for symnode references.

        Resolution strategy based on Origin type:
        1. BlockSizeOrigin -> Use pre-emitted ctx.block_size_ssa[canonical_block_id]
        2. Other origins -> Use shape_env.var_to_val for concrete value -> arith.constant
        """
        from helion._compiler.variable_origin import BlockSizeOrigin

        sym_val = node.meta.get("val")
        if sym_val is None:
            raise ValueError(f"No meta['val'] for symnode {node.name}")

        sym = sym_val._sympy_()

        host_function = self.ctx.bound_kernel.host_function
        origin_info = host_function.expr_to_origin.get(sym)
        origin = origin_info.origin if origin_info else None

        if isinstance(origin, BlockSizeOrigin):
            raw_block_id = origin.block_id
            canonical_id = self.ctx.resolve_block_id(raw_block_id)
            block_info = self.ctx.env.block_sizes[raw_block_id]

            if isinstance(block_info.size, int):
                ssa = self.mlir_output_helper.fresh(node.name.replace(".", "_"))
                self.mlir_output_helper.emit(f'{ssa} = arith.constant {block_info.size} : index')
                self.ctx.node_values[node.name] = ssa
                return ssa

            ssa = self.ctx.block_size_ssa.get(canonical_id)
            if ssa is None:
                raise ValueError(f"No block_size_ssa for block_id={canonical_id}")
            self.ctx.node_values[node.name] = ssa
            return ssa

        shape_env = self.ctx.bound_kernel.env.shape_env
        if sym in shape_env.var_to_val:
            concrete_val = int(shape_env.var_to_val[sym])
            ssa = self.mlir_output_helper.fresh(node.name.replace(".", "_"))
            self.mlir_output_helper.emit(f'{ssa} = arith.constant {concrete_val} : index')
            self.ctx.node_values[node.name] = ssa
            return ssa

        raise ValueError(
            f"Cannot resolve symbol {sym} for {node.name}. "
            f"Origin: {origin}, not in shape_env.var_to_val"
        )
    
    def visit_full(self, node: fx.Node) -> str:
        """Generate tensor.empty + linalg.fill for tensor initialization.

        Replaces the custom helion.full with standard MLIR dialects:
        - tensor.empty to create an uninitialized tensor
        - linalg.fill to fill it with the specified value

        For 0-rank tensors (empty shape), emits a bare scalar arith.constant
        instead, so downstream linalg.generic ops can capture it directly
        in their body rather than as an ins() tensor operand.

        For special values like -inf, we emit arith.constant with IEEE 754 hex
        representation.
        """
        import math

        shape_nodes = node.args[0]  # List of FX nodes or values
        fill_value = node.args[1] if len(node.args) > 1 else 0.0
        dtype = node.args[2] if len(node.args) > 2 else torch.float32

        # Special case: 0-rank tensor -> emit bare scalar arith.constant
        if len(shape_nodes) == 0:
            dtype_str = torch_dtype_to_mlir_element_type(dtype) if dtype else "f32"
            cst_ssa = self.mlir_output_helper.fresh("cst_scalar")
            if isinstance(fill_value, float) and (math.isinf(fill_value) or math.isnan(fill_value)):
                hex_val = self._get_hex_constant(fill_value, dtype_str)
                self.mlir_output_helper.emit(f'{cst_ssa} = arith.constant {hex_val} : {dtype_str}')
            else:
                self.mlir_output_helper.emit(f'{cst_ssa} = arith.constant {fill_value} : {dtype_str}')
            self.ctx.node_values[node.name] = cst_ssa
            self.ctx.node_types[node.name] = dtype_str  # "f16", NOT "tensor<f16>"
            return cst_ssa

        # Resolve shape to SSA values - integer literals need to become arith.constant
        shape_ssa = []
        for s in shape_nodes:
            if isinstance(s, fx.Node):
                shape_ssa.append(self.ctx.node_values.get(s.name, f"%{s.name}"))
            elif isinstance(s, int):
                # Integer literals need to be emitted as arith.constant
                const_ssa = self.mlir_output_helper.fresh("dim")
                self.mlir_output_helper.emit(f'{const_ssa} = arith.constant {s} : index')
                shape_ssa.append(const_ssa)
            else:
                raise RuntimeError(f"Unsupported shape type: {type(s)}")
        
        dtype_str = torch_dtype_to_mlir_element_type(dtype) if dtype else "f32"
        
        # Use FakeTensor metadata to determine tensor type with correct dimensions
        # This preserves concrete dimensions (e.g., 128 for head_dim) instead of using '?' for all
        fake_tensor = node.meta.get("val")
        if fake_tensor is not None and hasattr(fake_tensor, "shape"):
            tensor_type = self.ctx.compute_mlir_type_from_fake_tensor(fake_tensor)
        else:
            raise RuntimeError(f"Cannot compute MLIR type for node {node.name}")
        
        # Step 1: Emit tensor.empty
        # Only include SSA values for dynamic dimensions (?)
        # Parse tensor_type to find dynamic dimensions
        if 'tensor<' in tensor_type:
            type_content = tensor_type[tensor_type.find('<')+1 : tensor_type.rfind('>')]
            if 'x' in type_content:
                dims = type_content.split('x')[:-1]  # Exclude element type
                # Filter shape_ssa to only include values for dynamic dimensions
                dynamic_shape_ssa = [
                    ssa for ssa, dim in zip(shape_ssa, dims) if dim == '?'
                ]
            else:
                dynamic_shape_ssa = shape_ssa
        else:
            dynamic_shape_ssa = shape_ssa
        
        empty_ssa = self.mlir_output_helper.fresh("empty")
        shape_str = ", ".join(dynamic_shape_ssa)
        self.mlir_output_helper.emit(f'{empty_ssa} = tensor.empty({shape_str}) : {tensor_type}')
        
        # Step 2: Emit fill value constant
        fill_val_ssa = self.mlir_output_helper.fresh("fill_val")
        if isinstance(fill_value, float) and (math.isinf(fill_value) or math.isnan(fill_value)):
            hex_val = self._get_hex_constant(fill_value, dtype_str)
            self.mlir_output_helper.emit(f'{fill_val_ssa} = arith.constant {hex_val} : {dtype_str}')
        else:
            # Regular float value
            self.mlir_output_helper.emit(f'{fill_val_ssa} = arith.constant {fill_value} : {dtype_str}')
        
        # Step 3: Emit linalg.fill
        filled_ssa = self.mlir_output_helper.fresh("filled")
        self.mlir_output_helper.emit(
            f'{filled_ssa} = linalg.fill ins({fill_val_ssa} : {dtype_str}) '
            f'outs({empty_ssa} : {tensor_type}) -> {tensor_type}'
        )
        
        self.ctx.node_values[node.name] = filled_ssa
        self.ctx.node_types[node.name] = tensor_type
        
        # Track initial accumulator for phi
        self.ctx.initial_acc_ssa[node.name] = filled_ssa
        
        return filled_ssa
    
    def visit_aten_full(self, node: fx.Node) -> str:
        """Generate tensor.empty + linalg.fill for aten.full.default (from torch.full_like).
        
        Replaces the custom helion.full with standard MLIR dialects:
        - tensor.empty to create an uninitialized tensor
        - linalg.fill to fill it with the specified value
        
        aten.full.default has a different signature than helion's full:
        - args[0]: shape as list of FX nodes or ints
        - args[1]: fill_value
        - kwargs: dtype, layout, device, pin_memory
        """
        import math
        
        shape_nodes = node.args[0]  # List of FX nodes (e.g., block_size_0, block_size_1)
        fill_value = node.args[1] if len(node.args) > 1 else 0.0
        dtype = node.kwargs.get('dtype', torch.float32)
        
        # Resolve shape to SSA values - integer literals need to become arith.constant
        shape_ssa = []
        for s in shape_nodes:
            if isinstance(s, fx.Node):
                shape_ssa.append(self.ctx.node_values.get(s.name, f"%{s.name}"))
            elif isinstance(s, int):
                # Integer literals need to be emitted as arith.constant
                const_ssa = self.mlir_output_helper.fresh("dim")
                self.mlir_output_helper.emit(f'{const_ssa} = arith.constant {s} : index')
                shape_ssa.append(const_ssa)
            else:
                raise RuntimeError(f"Unsupported shape type: {type(s)}")
        
        dtype_str = torch_dtype_to_mlir_element_type(dtype) if dtype else "f32"
        
        # Use FakeTensor metadata to determine tensor type with correct dimensions
        # This preserves concrete dimensions (e.g., 128 for head_dim) instead of using '?' for all
        fake_tensor = node.meta.get("val")
        if fake_tensor is not None and hasattr(fake_tensor, "shape"):
            tensor_type = self.ctx.compute_mlir_type_from_fake_tensor(fake_tensor)
        else:
            raise RuntimeError(f"Cannot compute MLIR type for node {node.name}")
        
        # Step 1: Emit tensor.empty
        # Only include SSA values for dynamic dimensions (?)
        # Parse tensor_type to find dynamic dimensions
        if 'tensor<' in tensor_type:
            type_content = tensor_type[tensor_type.find('<')+1 : tensor_type.rfind('>')]
            if 'x' in type_content:
                dims = type_content.split('x')[:-1]  # Exclude element type
                # Filter shape_ssa to only include values for dynamic dimensions
                dynamic_shape_ssa = [
                    ssa for ssa, dim in zip(shape_ssa, dims) if dim == '?'
                ]
            else:
                dynamic_shape_ssa = shape_ssa
        else:
            dynamic_shape_ssa = shape_ssa
        
        empty_ssa = self.mlir_output_helper.fresh("empty")
        shape_str = ", ".join(dynamic_shape_ssa)
        self.mlir_output_helper.emit(f'{empty_ssa} = tensor.empty({shape_str}) : {tensor_type}')
        
        # Step 2: Emit fill value constant
        fill_val_ssa = self.mlir_output_helper.fresh("fill_val")
        if isinstance(fill_value, float) and (math.isinf(fill_value) or math.isnan(fill_value)):
            hex_val = self._get_hex_constant(fill_value, dtype_str)
            self.mlir_output_helper.emit(f'{fill_val_ssa} = arith.constant {hex_val} : {dtype_str}')
        else:
            # Regular float value
            self.mlir_output_helper.emit(f'{fill_val_ssa} = arith.constant {fill_value} : {dtype_str}')
        
        # Step 3: Emit linalg.fill
        filled_ssa = self.mlir_output_helper.fresh("filled")
        self.mlir_output_helper.emit(
            f'{filled_ssa} = linalg.fill ins({fill_val_ssa} : {dtype_str}) '
            f'outs({empty_ssa} : {tensor_type}) -> {tensor_type}'
        )
        
        self.ctx.node_values[node.name] = filled_ssa
        self.ctx.node_types[node.name] = tensor_type
        
        # Track initial accumulator for phi
        self.ctx.initial_acc_ssa[node.name] = filled_ssa
        
        return filled_ssa
    
    def visit_dot(self, node: fx.Node) -> str:
        """Lower Helion dot op to linalg.matmul."""
        if len(node.args) < 3:
            raise RuntimeError(f"dot expects (lhs, rhs, acc, ...), got args={node.args}")

        lhs, rhs, acc = node.args[0], node.args[1], node.args[2]
        if not isinstance(lhs, fx.Node) or not isinstance(rhs, fx.Node) or not isinstance(acc, fx.Node):
            raise RuntimeError(f"dot expects FX node operands, got {type(lhs)}, {type(rhs)}, {type(acc)}")

        lhs_ssa = self.ctx.node_values.get(lhs.name, f"%{lhs.name}")
        rhs_ssa = self.ctx.node_values.get(rhs.name, f"%{rhs.name}")
        acc_ssa = self.ctx.node_values.get(acc.name, f"%{acc.name}")

        lhs_type = self._get_tensor_type(lhs)
        rhs_type = self._get_tensor_type(rhs)
        acc_type = self._get_tensor_type(acc)

        result = self.mlir_output_helper.fresh("dot")
        self.mlir_output_helper.emit(
            f"{result} = linalg.matmul "
            f"ins({lhs_ssa}, {rhs_ssa} : {lhs_type}, {rhs_type}) "
            f"outs({acc_ssa} : {acc_type}) -> {acc_type}"
        )

        self.ctx.node_values[node.name] = result
        self.ctx.node_types[node.name] = acc_type
        return result

    def _is_python_index_arith_target(self, target: Any) -> bool:
        """Return True for Python builtins/operator arithmetic on scalar indices."""
        name = getattr(target, "__name__", None)
        module = getattr(target, "__module__", "")
        return (
            name in {"add", "sub", "mul", "floordiv"}
            and module in {"builtins", "operator", "_operator"}
        )

    def _emit_index_constant(self, value: int) -> str:
        ssa = self.mlir_output_helper.fresh("idx")
        self.mlir_output_helper.emit(f"{ssa} = arith.constant {int(value)} : index")
        return ssa

    def _emit_index_divui(self, lhs: str, rhs: str, *, hint: str = "idx_div") -> str:
        result = self.mlir_output_helper.fresh(hint)
        self.mlir_output_helper.emit(f"{result} = arith.divui {lhs}, {rhs} : index")
        return result

    def _emit_index_ceildivui(
        self,
        lhs: str,
        rhs: str,
        *,
        hint: str = "idx_ceildiv",
    ) -> str:
        result = self.mlir_output_helper.fresh(hint)
        self.mlir_output_helper.emit(f"{result} = arith.ceildivui {lhs}, {rhs} : index")
        return result

    def _emit_index_addi(self, lhs: str, rhs: str, *, hint: str = "idx_add") -> str:
        result = self.mlir_output_helper.fresh(hint)
        self.mlir_output_helper.emit(f"{result} = arith.addi {lhs}, {rhs} : index")
        return result

    def _emit_index_subi(self, lhs: str, rhs: str, *, hint: str = "idx_sub") -> str:
        result = self.mlir_output_helper.fresh(hint)
        self.mlir_output_helper.emit(f"{result} = arith.subi {lhs}, {rhs} : index")
        return result

    def _emit_index_muli(self, lhs: str, rhs: str, *, hint: str = "idx_mul") -> str:
        result = self.mlir_output_helper.fresh(hint)
        self.mlir_output_helper.emit(f"{result} = arith.muli {lhs}, {rhs} : index")
        return result

    def _emit_index_minui(self, lhs: str, rhs: str, *, hint: str = "idx_min") -> str:
        pred = self.mlir_output_helper.fresh("idx_lt")
        self.mlir_output_helper.emit(f"{pred} = arith.cmpi ult, {lhs}, {rhs} : index")
        result = self.mlir_output_helper.fresh(hint)
        self.mlir_output_helper.emit(f"{result} = arith.select {pred}, {lhs}, {rhs} : index")
        return result

    def _get_block_loop_iv(self, canonical_block_id: int) -> str:
        return self.ctx.loop_resolver.get_block_loop_iv(canonical_block_id)

    def _get_active_loop_bounds(self, canonical_block_id: int) -> tuple[str, str] | None:
        if canonical_block_id in self.loop_bounds:
            return self.loop_bounds.get(canonical_block_id)
        return self.ctx.loop_resolver.get_active_loop_bounds(canonical_block_id)

    def _emit_block_offset(
        self,
        canonical_block_id: int,
        block_size_ssa: str,
        *,
        hint: str = "offset",
    ) -> str:
        iv_ssa = self._get_block_loop_iv(canonical_block_id)
        offset_ssa = self._emit_index_muli(iv_ssa, block_size_ssa, hint=hint)
        bounds = self._get_active_loop_bounds(canonical_block_id)
        if bounds is None:
            return offset_ssa
        lb_ssa, _ = bounds
        return self._emit_index_addi(lb_ssa, offset_ssa, hint=f"{hint}_base")

    def _emit_block_end(
        self,
        canonical_block_id: int,
        block_size_ssa: str,
        *,
        hint: str = "tile_end",
    ) -> str:
        start_ssa = self._emit_block_offset(
            canonical_block_id,
            block_size_ssa,
            hint=f"{hint}_start",
        )
        naive_end_ssa = self._emit_index_addi(
            start_ssa,
            block_size_ssa,
            hint=f"{hint}_naive",
        )
        bounds = self._get_active_loop_bounds(canonical_block_id)
        if bounds is None or self.ctx.assume_divisible_tiles:
            return naive_end_ssa
        _, ub_ssa = bounds
        return self._emit_index_minui(naive_end_ssa, ub_ssa, hint=hint)

    def _emit_block_extent(
        self,
        canonical_block_id: int,
        block_size_ssa: str,
        *,
        hint: str = "tile_extent",
    ) -> str:
        if self.ctx.assume_divisible_tiles:
            return block_size_ssa
        bounds = self._get_active_loop_bounds(canonical_block_id)
        if bounds is None:
            return block_size_ssa
        start_ssa = self._emit_block_offset(
            canonical_block_id,
            block_size_ssa,
            hint=f"{hint}_start",
        )
        end_ssa = self._emit_block_end(
            canonical_block_id,
            block_size_ssa,
            hint=f"{hint}_end",
        )
        return self._emit_index_subi(end_ssa, start_ssa, hint=hint)

    def _as_index_ssa(self, value: Any) -> str:
        """Materialize an index-typed SSA value from literals/SymInt/FX nodes."""
        if isinstance(value, fx.Node):
            ssa = self.ctx.node_values.get(value.name)
            if ssa is None:
                visited = self.visit_node(value)
                ssa = visited if isinstance(visited, str) else None
            if ssa is None:
                raise RuntimeError(f"Unable to resolve index value for node {value.name}")
            if isinstance(ssa, str) and not ssa.startswith("%"):
                try:
                    return self._emit_index_constant(int(ssa))
                except ValueError:
                    raise RuntimeError(f"Unsupported non-SSA index value: {ssa}") from None
            return ssa

        if isinstance(value, bool):
            return self._emit_index_constant(1 if value else 0)
        if isinstance(value, int):
            return self._emit_index_constant(value)

        if hasattr(value, "_sympy_"):
            resolved, _ = self.resolve_dimension(value, 0)
            if resolved is None:
                raise RuntimeError(f"Cannot resolve symbolic index value: {value}")
            if resolved.startswith("%"):
                return resolved
            return self._emit_index_constant(int(resolved))

        if isinstance(value, str):
            if value.startswith("%"):
                return value
            try:
                return self._emit_index_constant(int(value))
            except ValueError:
                raise RuntimeError(f"Unsupported index string value: {value}") from None

        raise RuntimeError(f"Unsupported index operand type: {type(value)} ({value})")

    def visit_python_index_arith(self, node: fx.Node) -> str:
        """Lower Python/operator scalar arithmetic to index ops."""
        target = node.target
        op_name = getattr(target, "__name__", None)
        if op_name not in {"add", "sub", "mul", "floordiv"}:
            raise RuntimeError(f"Unsupported python index op: {target}")
        if len(node.args) != 2:
            raise RuntimeError(f"Expected binary index op, got args={node.args}")

        lhs = self._as_index_ssa(node.args[0])
        rhs = self._as_index_ssa(node.args[1])

        result = self.mlir_output_helper.fresh(node.name.replace(".", "_"))
        if op_name == "add":
            self.mlir_output_helper.emit(f"{result} = arith.addi {lhs}, {rhs} : index")
        elif op_name == "sub":
            self.mlir_output_helper.emit(f"{result} = arith.subi {lhs}, {rhs} : index")
        elif op_name == "mul":
            self.mlir_output_helper.emit(f"{result} = arith.muli {lhs}, {rhs} : index")
        else:
            result = self._emit_index_divui(lhs, rhs, hint=node.name.replace(".", "_"))

        self.ctx.node_values[node.name] = result
        self.ctx.node_types[node.name] = "index"
        return result

    def visit_tile_index_add(self, node: fx.Node) -> str | None:
        """Lower aten.add(tile_index, base) into a contiguous index-range base."""
        extracted = self._extract_tile_index_add(node)
        if extracted is None:
            return None

        block_id, base = extracted
        base_ssa = self._as_index_ssa(base)
        block_size_ssa, block_size_is_static = self._get_block_size_value(block_id)
        if block_size_is_static:
            block_size_ssa = self._emit_index_constant(int(block_size_ssa))

        # Preserve tile-start contribution when rewriting range indexing.
        # Dropping tile_start is incorrect for multi-iteration tiled dims.
        tile_start_ssa = self._emit_block_offset(block_id, block_size_ssa, hint="tile_idx_start")
        combined_base_ssa = self._emit_index_addi(tile_start_ssa, base_ssa, hint="tile_idx_base")

        self.range_index_block_ids[node.name] = block_id
        self.ctx.node_values[node.name] = combined_base_ssa
        self.ctx.node_types[node.name] = "index"
        return combined_base_ssa

    def _derive_loop_trip_info_from_for_loop_args(
        self, node: fx.Node, canonical_block_id: int
    ) -> tuple[str, str, str] | None:
        """Primary path: derive [lb, ub) trip count information from FX _for_loop bounds."""
        if len(node.args) < 3:
            return None

        lower_bounds = node.args[1]
        upper_bounds = node.args[2]
        if not isinstance(lower_bounds, (list, tuple)) or not isinstance(
            upper_bounds, (list, tuple)
        ):
            return None
        if len(lower_bounds) != 1 or len(upper_bounds) != 1:
            return None

        lb = lower_bounds[0]
        ub = upper_bounds[0]

        # Optional explicit step list (if present in FX args) must be unit.
        if len(node.args) > 4:
            steps = node.args[4]
            step = steps[0] if isinstance(steps, (list, tuple)) and len(steps) == 1 else steps
            if not isinstance(step, int) or step != 1:
                raise ValueError(
                    f"Only unit-step block loops are supported for now; got step {step!r}"
                )

        lb_ssa = self._as_index_ssa(lb)
        ub_ssa = self._as_index_ssa(ub)
        block_info = self.ctx.env.block_sizes[canonical_block_id]
        if isinstance(block_info.size, int):
            tile_size_ssa = self._emit_index_constant(int(block_info.size))
        else:
            tile_size_ssa = self.ctx.block_size_ssa.get(canonical_block_id)
            if tile_size_ssa is None:
                return None

        extent_ssa = (
            ub_ssa
            if isinstance(lb, int) and lb == 0
            else self._emit_index_subi(ub_ssa, lb_ssa, hint="loop_extent")
        )
        trip_count_ssa = self._emit_index_ceildivui(
            extent_ssa,
            tile_size_ssa,
            hint="trip_count",
        )
        return (trip_count_ssa, lb_ssa, ub_ssa)

    def visit_for_loop(self, node: fx.Node) -> str:
        """Generate scf.for and visit the inner ForLoopGraphInfo."""
        graph_id = node.args[0]
        args = node.args[3]   # [acc]

        # Get the ForLoopGraphInfo
        for_graph = self.ctx.graphs.get(graph_id)
        if for_graph is None:
            raise ValueError(f"ForLoopGraphInfo with graph_id={graph_id} not registered")

        # Get block_ids from the graph (resolve through alias map)
        block_ids = getattr(for_graph, "block_ids", [self.loop_depth])
        block_id = self.ctx.resolve_block_id(block_ids[0] if block_ids else 2)

        # Primary path: derive trip count from _for_loop FX bounds.
        # Fallback path: use pre-computed metadata trip count.
        loop_trip_info = self._derive_loop_trip_info_from_for_loop_args(node, block_id)
        if loop_trip_info is not None:
            trip_count_ssa, lb_ssa, ub_ssa = loop_trip_info
        else:
            trip_count_ssa = self.ctx.reduction_trip_counts.get(block_id)
            lb_ssa = None
            ub_ssa = None
        if trip_count_ssa is None:
            raise ValueError("No trip count found for loop")

        # ---------------------------------------------------------------------
        # Collect all arguments and classify them using phi-node analysis:
        # - Read-only args: passed to the loop but NOT yielded back
        # - Loop-carried args (iter_args): yielded and updated each iteration
        # ---------------------------------------------------------------------
        iter_arg_names = set()
        for user in node.users:
            if user.target is operator.getitem:
                for phi_user in user.users:
                    if hasattr(phi_user, "target") and phi_user.target is hl_tracing_ops._phi:
                        init_val = phi_user.args[0]
                        if isinstance(init_val, fx.Node):
                            iter_arg_names.add(init_val.name)

        all_args_info = []
        for i, a in enumerate(args):
            if isinstance(a, fx.Node):
                ssa = self.ctx.node_values.get(a.name, f"%{a.name}")
                all_args_info.append((f"acc_iter{i}", ssa, a.name))
            else:
                all_args_info.append((f"acc_iter{i}", str(a), None))

        iter_args_info = []
        for info_tuple in all_args_info:
            name, ssa, fx_name = info_tuple
            if fx_name and fx_name in iter_arg_names:
                iter_args_info.append((name, ssa, fx_name))

        iter_args_types = []
        for _, _, fx_name in iter_args_info:
            if fx_name and fx_name in self.ctx.node_types:
                iter_args_types.append(self.ctx.node_types[fx_name])
            else:
                fx_node = next(
                    (a for a in args if isinstance(a, fx.Node) and a.name == fx_name),
                    None,
                )
                if fx_node is not None and "val" in fx_node.meta:
                    inferred_type = self.ctx.compute_mlir_type_from_fake_tensor(
                        fx_node.meta["val"]
                    )
                    self.ctx.node_types[fx_name] = inferred_type
                    iter_args_types.append(inferred_type)
                else:
                    raise RuntimeError(f"Cannot compute MLIR type for node {fx_name}")

        iv = f"%iv_block_{block_id}"
        result = self.mlir_output_helper.fresh(f"for_result_{graph_id}")

        iter_args_parts = [f"%{name} = {ssa}" for name, ssa, _ in iter_args_info]
        iter_args_str = ", ".join(iter_args_parts)
        result_types = ", ".join(iter_args_types)

        num_results = len(iter_args_info)
        result_binding = f"{result}:{num_results}" if num_results > 1 else result

        c0 = self._emit_index_constant(0)
        c1 = self._emit_index_constant(1)
        self.mlir_output_helper.emit(
            f"{result_binding} = scf.for {iv} = {c0} to {trip_count_ssa} step {c1} "
            f"iter_args({iter_args_str}) -> ({result_types}) {{"
        )
        yield_op = "scf.yield"
        self.mlir_output_helper.push()

        old_loop_iter_args = self.loop_iter_args.copy()

        iter_arg_idx = 0
        for i, a in enumerate(args):
            placeholder_name = f"arg{i}_1"
            if isinstance(a, fx.Node) and a.name in iter_arg_names:
                iter_name = iter_args_info[iter_arg_idx][0]
                self._propagate_metadata(a.name, placeholder_name, ssa_override=f"%{iter_name}", is_placeholder=True)
                iter_arg_idx += 1
            elif isinstance(a, fx.Node):
                self._propagate_metadata(a.name, placeholder_name, is_placeholder=True)

        self.loop_depth += 1

        old_block_id = self.current_block_id
        self.current_block_id = block_id
        saved_loop_bounds = self.loop_bounds.copy()
        if lb_ssa is not None and ub_ssa is not None:
            self.loop_bounds[block_id] = (lb_ssa, ub_ssa)

        # ForLoopGraphInfo nodes are in a separate FX graph and may reuse node
        # names from the parent graph (e.g. tile_begin, mul_1). Keep a local
        # scope so inner-body values/types do not leak back to the parent graph.
        with self.ctx.push_graph_scope():
            saved_range_index_block_ids = self.range_index_block_ids.copy()
            self.visit_graph(for_graph)
            loop_result_snapshot = self.current_loop_result
            loop_result_values_snapshot = self.ctx.loop_result_values.copy()
            self.range_index_block_ids = saved_range_index_block_ids

        self.current_loop_result = loop_result_snapshot

        self.current_block_id = old_block_id
        self.loop_depth -= 1
        self.loop_iter_args = old_loop_iter_args
        self.loop_bounds = saved_loop_bounds

        if isinstance(self.current_loop_result, list) and len(self.current_loop_result) > 1:
            yield_values = ", ".join(self.current_loop_result)
            yield_types = ", ".join(iter_args_types)
            self.mlir_output_helper.emit(f"{yield_op} {yield_values} : {yield_types}")
        else:
            yield_value = (
                self.current_loop_result[0]
                if isinstance(self.current_loop_result, list)
                else (self.current_loop_result or f"%{iter_args_info[0][0]}")
            )
            yield_type = iter_args_types[0] if iter_args_types else "tensor<?x?xf32>"
            self.mlir_output_helper.emit(f"{yield_op} {yield_value} : {yield_type}")

        self.mlir_output_helper.pop()
        self.mlir_output_helper.emit("}")

        self.ctx.node_values[node.name] = result
        if isinstance(self.current_loop_result, list) and len(self.current_loop_result) > 1:
            self.ctx.loop_result_values = loop_result_values_snapshot
            self.ctx.record_loop_result(
                node.name,
                result,
                count=len(self.current_loop_result),
            )

        return result

    
    def visit_phi(self, node: fx.Node) -> str:
        """Handle phi nodes for loop-carried value merging.
        
        For Helion Device IR, _phi merges the initial value with the loop result.
        Since MLIR loop iter_args already handle merge semantics,
        we simply pass through the loop result SSA.
        
        Pattern detected:
        - lhs: initial accumulator (e.g., from hl.full)
        - rhs: getitem of _for_loop result
        
        If this pattern is detected, we just use the loop result SSA.
        Otherwise, raise an error (unsupported phi pattern).
        """
        import helion.language._tracing_ops as hl_tracing_ops
        import operator
        
        lhs = node.args[0]  # Initial value
        rhs = node.args[1]  # Loop result (getitem)
        
        # Check if this is a loop merge pattern
        is_loop_merge = False
        loop_result_ssa = None
        
        if isinstance(rhs, fx.Node):
            # Check if rhs is a getitem on _for_loop
            if rhs.target is operator.getitem:
                source = rhs.args[0]
                if isinstance(source, fx.Node) and source.target is hl_tracing_ops._for_loop:
                    # This is a getitem(_for_loop, index) - classic loop merge pattern
                    is_loop_merge = True
                    loop_result_ssa = self.ctx.node_values.get(rhs.name)
                    
                    # Verify lhs is in the _for_loop's args (iter_args)
                    for_loop_args = source.args[3] if len(source.args) > 3 else []
                    if isinstance(lhs, fx.Node) and lhs in for_loop_args:
                        # Confirmed: lhs is an iter_arg of the loop
                        pass
                    # Even if not in args, treat as loop merge for now
        
        if is_loop_merge and loop_result_ssa is not None:
            # No helion.phi needed - MLIR's loop iter_args handle the merge
            # Just pass through the loop result
            self.ctx.node_values[node.name] = loop_result_ssa
            if isinstance(rhs, fx.Node):
                self.ctx.node_types[node.name] = self._get_tensor_type(rhs)
            return loop_result_ssa
        
        # Unsupported phi pattern - raise error as requested
        raise ValueError(
            f"Unsupported phi pattern in {node.name}: "
            f"lhs={lhs}, rhs={rhs}. "
            "Only loop-carried value merging (_phi(acc, getitem(_for_loop, i))) is supported."
        )
    
    def visit_new_var(self, node: fx.Node) -> str:
        """Pass through _new_var nodes (just forward the input value)."""
        arg = node.args[0]
        if isinstance(arg, fx.Node):
            self._propagate_metadata(arg.name, node.name)
            return self.ctx.node_values[node.name]
        
        ssa = str(arg)
        self.ctx.node_values[node.name] = ssa
        return ssa
    
    def visit_host_tensor(self, node: fx.Node) -> str:
        """Map host tensor reference to function argument.
        
        All host tensors (both kernel args and derived tensors like views)
        are pre-registered as function parameters during MLIR generation.
        This method simply looks them up - no helion.host_ref emission needed.
        """
        tensor_name = node.args[0]  # 'x', 'y', 'out', 'q_view', etc.
        
        # Look up in pre-registered host tensors
        ssa = self.ctx.host_tensors.get(tensor_name)
        if ssa is None:
            raise ValueError(
                f"Host tensor '{tensor_name}' not found in pre-registered host_tensors. "
                f"Available: {list(self.ctx.host_tensors.keys())}. "
                "All _host_tensor calls should map to function parameters."
            )
        
        self.ctx.node_values[node.name] = ssa
        return ssa

    
    def visit_sym_size(self, node: fx.Node) -> str:
        """Handle aten.sym_size.int - get dimension size from a tensor.
        
        In ForLoopGraphInfo, sym_size_int is used to get the dimension size
        of a tensor argument. Optimized to use known dimension values from
        FakeTensor metadata instead of emitting tensor.dim when possible.
        """
        tensor_node = node.args[0]
        dim = node.args[1]
        
        # Get the tensor SSA value
        # First, try to resolve from loop_iter_args (for read-only args passed to for_loop)
        if isinstance(tensor_node, fx.Node):
            tensor_ssa = self.loop_iter_args.get(tensor_node.name)
            if tensor_ssa is None:
                tensor_ssa = self.ctx.node_values.get(tensor_node.name, f"%{tensor_node.name}")
        else:
            tensor_ssa = str(tensor_node)
        
        # Try to resolve dimension from FakeTensor metadata
        if isinstance(tensor_node, fx.Node):
            fake_tensor = tensor_node.meta.get("val")
            if fake_tensor is not None and hasattr(fake_tensor, 'ndim') and dim < fake_tensor.ndim:
                dim_size = fake_tensor.size(dim)
                value_str, is_static = self.resolve_dimension(dim_size, dim)
                if value_str is not None:
                    # We have a known value - use it directly
                    self.ctx.node_values[node.name] = value_str
                    return value_str
        
        # Fallback: emit tensor.dim to get the dimension size
        if not isinstance(tensor_node, fx.Node):
            raise RuntimeError(f"Expected fx.Node for tensor_node, got {type(tensor_node)}: {tensor_node}")
        tensor_type = self._get_tensor_type(tensor_node)
        
        dim_idx_ssa = self.mlir_output_helper.fresh("dim_idx")
        self.mlir_output_helper.emit(f'{dim_idx_ssa} = arith.constant {dim} : index')
        
        result_ssa = self.mlir_output_helper.fresh("dim_size")
        self.mlir_output_helper.emit(f'{result_ssa} = tensor.dim {tensor_ssa}, {dim_idx_ssa} : {tensor_type}')
        
        self.ctx.node_values[node.name] = result_ssa
        return result_ssa
    
    def visit_load(self, node: fx.Node) -> str:
        """Generate memref.subview + bufferization.to_tensor for tile loading.
        
        Host tensors are memref types. This method:
        1. Emits memref.subview to get a view into the tile
        2. Emits bufferization.to_tensor to convert to tensor for linalg ops
        
        Uses inline literals for known constants (e.g., [0, %offset][%size, 128][1, 1])
        instead of emitting arith.constant for every value.
        """
        from helion._compiler.variable_origin import BlockSizeOrigin
        
        tensor_node = node.args[0]
        indices = node.args[1]  # [sym_size_int, block_size_2] or [block_size_0, block_size_1, slice(...)]
        
        memref_ssa = self.ctx.node_values.get(tensor_node.name, f"%{tensor_node.name}")
        memref_type = self._get_tensor_type(tensor_node)  # Now returns memref type for host tensors
        
        # Get output FakeTensor shape from node metadata to determine tile sizes
        output_fake_tensor = node.meta.get("val")
        host_function = self.ctx.bound_kernel.host_function
        shape_env = self.ctx.bound_kernel.env.shape_env
        
        def resolve_symint(sym_int, dim_hint: int) -> tuple[str, bool]:
            """Resolve a SymInt to its value (uses canonical block IDs)."""
            sym = sym_int._sympy_()
            origin_info = host_function.expr_to_origin.get(sym)
            origin = origin_info.origin if origin_info else None
            
            if isinstance(origin, BlockSizeOrigin):
                raw_id = origin.block_id
                canonical_id = self.ctx.resolve_block_id(raw_id)
                block_info = self.ctx.env.block_sizes[raw_id]
                
                if isinstance(block_info.size, int):
                    return (str(block_info.size), True)
                
                ssa = self.ctx.block_size_ssa.get(canonical_id)
                if ssa is not None:
                    return (ssa, False)
            
            if sym in shape_env.var_to_val:
                concrete_val = int(shape_env.var_to_val[sym])
                return (str(concrete_val), True)
            
            raise ValueError(f"Cannot resolve SymInt {sym} for dimension {dim_hint}")
        
        offsets = []
        sizes = []
        output_dim_sizes = []
        retained_dim_positions = []
        
        src_dims = self._parse_plain_memref_dimensions(memref_type, "load")
        
        for i, idx in enumerate(indices):
            if isinstance(idx, slice):
                self._validate_full_unit_slice(idx, "load", i)
                # Full dimension slice - offset=0 (static), size comes from memref dim
                offsets.append(("0", True))
                
                # Get dimension size from source memref
                src_dim_static = None
                if i < len(src_dims):
                    if src_dims[i] != '?':
                        try:
                            src_dim_static = int(src_dims[i])
                        except ValueError:
                            pass
                
                if src_dim_static is not None:
                    # Static dimension - use inline literal
                    sizes.append((str(src_dim_static), True))
                else:
                    # Dynamic - need memref.dim
                    dim_idx_ssa = self.mlir_output_helper.fresh("dim_idx")
                    self.mlir_output_helper.emit(f'{dim_idx_ssa} = arith.constant {i} : index')
                    dim_ssa = self.mlir_output_helper.fresh("dim")
                    self.mlir_output_helper.emit(f'{dim_ssa} = memref.dim {memref_ssa}, {dim_idx_ssa} : {memref_type}')
                    sizes.append((dim_ssa, False))
                output_dim_sizes.append(src_dim_static)
                retained_dim_positions.append(i)
                
            elif isinstance(idx, fx.Node):
                # Get the SSA value for this index (tile size)
                idx_ssa = self.ctx.node_values.get(idx.name, f"%{idx.name}")

                # tile_index + base pattern: contiguous range [base, base + block_size)
                range_block_id = self.range_index_block_ids.get(idx.name)
                if range_block_id is not None:
                    size_val, size_is_static = self._get_block_size_value(range_block_id)
                    offsets.append((idx_ssa, not idx_ssa.startswith("%")))
                    sizes.append((size_val, size_is_static))
                    if size_is_static:
                        try:
                            output_dim_sizes.append(int(size_val))
                        except ValueError:
                            output_dim_sizes.append(None)
                    else:
                        output_dim_sizes.append(None)
                    retained_dim_positions.append(i)
                    continue
                
                # Check if this index has a BlockSizeOrigin - if so, it represents a tile size
                # and we need to compute offset as iv * block_size
                block_id = self._try_get_block_id_from_node(idx)
                
                if block_id is not None:
                    if self._is_singleton_block(block_id):
                        # Scalar indexing for singleton blocks (size == 1): rank-reduce.
                        offsets.append((idx_ssa, not idx_ssa.startswith("%")))
                        sizes.append(("1", True))
                    else:
                        # Range indexing for regular tile blocks.
                        offset_ssa = self._emit_block_offset(block_id, idx_ssa, hint="offset")
                        size_ssa = self._emit_block_extent(block_id, idx_ssa, hint="tile_extent")
                        offsets.append((offset_ssa, False))
                        size_is_static = not size_ssa.startswith("%")
                        sizes.append((size_ssa, size_is_static))
                        if size_is_static:
                            try:
                                output_dim_sizes.append(int(size_ssa))
                            except ValueError:
                                output_dim_sizes.append(None)
                        else:
                            output_dim_sizes.append(None)
                        retained_dim_positions.append(i)
                else:
                    # Generic scalar index - rank-reduce this dimension.
                    offsets.append((idx_ssa, False))
                    sizes.append(("1", True))
                    
            elif isinstance(idx, int):
                # Integer scalar index - rank-reduce this dimension.
                offsets.append((str(idx), True))
                sizes.append(("1", True))
            else:
                raise RuntimeError(f"Unsupported index type: {type(idx)}")
        
        # Extract dtype from memref type
        content = memref_type[memref_type.find('<')+1 : memref_type.rfind('>')]
        if 'x' in content:
            dtype_str = content.split('x')[-1]
        else:
            dtype_str = content
        
        # Build output dimensions for both memref subview and tensor result
        output_dims = []
        for dim_size in output_dim_sizes:
            if dim_size is not None:
                output_dims.append(str(dim_size))
            else:
                output_dims.append("?")
        
        # Construct subview result type (memref) with strided layout
        # For memref.subview with dynamic offsets from a static layout source,
        # the result must have strided layout with offset: ?
        
        # Compute strides from source memref dimensions (row-major)
        # For source memref<D0xD1xD2xf32>, strides are [D1*D2, D2, 1]
        def compute_strides_from_dims(dims: list[str]) -> list[str]:
            """Compute row-major strides from dimension strings (dynamic dims -> '?')."""
            strides: list[str] = []
            running_product: int | None = 1
            for dim in reversed(dims):
                strides.append(str(running_product) if running_product is not None else "?")
                try:
                    dim_int = int(dim)
                except ValueError:
                    running_product = None
                else:
                    if running_product is not None:
                        running_product *= dim_int
            return list(reversed(strides))
        
        src_strides = compute_strides_from_dims(src_dims)
        has_dynamic_offset = any(not is_static for _, is_static in offsets)
        if output_dims:
            if has_dynamic_offset:
                # Need strided layout with offset:? whenever offsets are dynamic.
                # For rank-reduced subviews, keep only the retained source strides.
                if retained_dim_positions:
                    layout_strides = [src_strides[p] for p in retained_dim_positions]
                else:
                    layout_strides = src_strides
                strides_str = ", ".join(layout_strides)
                subview_type = f"memref<{'x'.join(output_dims)}x{dtype_str}, strided<[{strides_str}], offset: ?>>"
            else:
                subview_type = f"memref<{'x'.join(output_dims)}x{dtype_str}>"
        else:
            if has_dynamic_offset:
                subview_type = f"memref<{dtype_str}, strided<[], offset: ?>>"
            else:
                subview_type = f"memref<{dtype_str}>"
        
        # Construct tensor result type
        if output_dims:
            tensor_type = f"tensor<{'x'.join(output_dims)}x{dtype_str}>"
        else:
            tensor_type = f"tensor<{dtype_str}>"
        
        self.ctx.node_types[node.name] = tensor_type
        
        # Format offsets, sizes, strides - using inline literals or SSA values
        offsets_str = ", ".join(v for v, _ in offsets)
        sizes_str = ", ".join(v for v, _ in sizes)
        strides_str = ", ".join(["1"] * len(indices))
        
        # Emit memref.subview
        subview_ssa = self.mlir_output_helper.fresh("subview")
        self.mlir_output_helper.emit(
            f'{subview_ssa} = memref.subview {memref_ssa}[{offsets_str}][{sizes_str}][{strides_str}] : '
            f'{memref_type} to {subview_type}'
        )
        
        # Emit bufferization.to_tensor to convert memref view to tensor
        result = self.mlir_output_helper.fresh("tile")
        self.mlir_output_helper.emit(
            f'{result} = bufferization.to_tensor {subview_ssa} : {subview_type} to {tensor_type}'
        )
        
        self.ctx.node_values[node.name] = result
        return result
    
    def visit_store(self, node: fx.Node) -> str:
        """Generate memref.subview + bufferization.to_memref + memref.copy for tile storing.
        
        Host tensors are memref types. This method:
        1. Emits memref.subview to get a view into the destination tile
        2. Emits bufferization.to_memref to convert source tensor to memref
        3. Emits memref.copy to copy data into the subview (in-place mutation)
        
        Uses inline literals for known constants (e.g., [0, %offset][%size, 128][1, 1])
        instead of emitting arith.constant for every value.
        """
        from helion._compiler.variable_origin import BlockSizeOrigin
        
        tensor_node = node.args[0]
        indices = node.args[1]
        value = node.args[2]
        
        memref_ssa = self.ctx.node_values.get(tensor_node.name, f"%{tensor_node.name}")
        memref_type = self._get_tensor_type(tensor_node)  # Now returns memref type for host tensors
        
        if not isinstance(value, fx.Node):
            raise RuntimeError(f"Expected fx.Node for value, got {type(value)}: {value}")
        value_ssa = self.ctx.node_values.get(value.name, f"%{value.name}")
        value_type = self._get_tensor_type(value)
        
        # Get FakeTensor from value node to determine tile sizes
        value_fake_tensor = value.meta.get("val") if isinstance(value, fx.Node) else None
        host_function = self.ctx.bound_kernel.host_function
        shape_env = self.ctx.bound_kernel.env.shape_env
        
        def resolve_symint(sym_int, dim_hint: int) -> tuple[str, bool]:
            """Resolve a SymInt to its value (uses canonical block IDs)."""
            sym = sym_int._sympy_()
            origin_info = host_function.expr_to_origin.get(sym)
            origin = origin_info.origin if origin_info else None
            
            if isinstance(origin, BlockSizeOrigin):
                raw_id = origin.block_id
                canonical_id = self.ctx.resolve_block_id(raw_id)
                block_info = self.ctx.env.block_sizes[raw_id]
                
                if isinstance(block_info.size, int):
                    return (str(block_info.size), True)
                
                ssa = self.ctx.block_size_ssa.get(canonical_id)
                if ssa is not None:
                    return (ssa, False)
            
            if sym in shape_env.var_to_val:
                concrete_val = int(shape_env.var_to_val[sym])
                return (str(concrete_val), True)
            
            raise ValueError(f"Cannot resolve SymInt {sym} for dimension {dim_hint}")
        
        offsets = []
        sizes = []
        output_dim_sizes = []
        retained_dim_positions = []
        
        def parse_type_dimensions(type_str: str) -> list[str]:
            if ('<' in type_str and '>' in type_str):
                content = type_str[type_str.find('<')+1 : type_str.rfind('>')]
                if 'x' in content:
                    return content.split('x')[:-1]
            return []
        
        src_dims = parse_type_dimensions(value_type)
        
        for i, idx in enumerate(indices):
            if isinstance(idx, slice):
                self._validate_full_unit_slice(idx, "store", i)
                # Full dimension slice - offset=0 (static), size comes from value tensor dim
                offsets.append(("0", True))
                
                # Check if value tensor's dimension is static
                src_dim_static = None
                if i < len(src_dims):
                    if src_dims[i] != '?':
                        try:
                            src_dim_static = int(src_dims[i])
                        except ValueError:
                            pass
                
                if src_dim_static is not None:
                    # Static dimension - use inline literal
                    sizes.append((str(src_dim_static), True))
                else:
                    # Dynamic - need tensor.dim on value tensor
                    dim_idx_ssa = self.mlir_output_helper.fresh("dim_idx")
                    self.mlir_output_helper.emit(f'{dim_idx_ssa} = arith.constant {i} : index')
                    dim_ssa = self.mlir_output_helper.fresh("dim")
                    self.mlir_output_helper.emit(f'{dim_ssa} = tensor.dim {value_ssa}, {dim_idx_ssa} : {value_type}')
                    sizes.append((dim_ssa, False))
                output_dim_sizes.append(src_dim_static)
                retained_dim_positions.append(i)
                
            elif isinstance(idx, fx.Node):
                idx_ssa = self.ctx.node_values.get(idx.name, f"%{idx.name}")

                # tile_index + base pattern: contiguous range [base, base + block_size)
                range_block_id = self.range_index_block_ids.get(idx.name)
                if range_block_id is not None:
                    size_val, size_is_static = self._get_block_size_value(range_block_id)
                    offsets.append((idx_ssa, not idx_ssa.startswith("%")))
                    sizes.append((size_val, size_is_static))
                    if size_is_static:
                        try:
                            output_dim_sizes.append(int(size_val))
                        except ValueError:
                            output_dim_sizes.append(None)
                    else:
                        output_dim_sizes.append(None)
                    retained_dim_positions.append(i)
                    continue
                
                # Check if this index has a BlockSizeOrigin - if so, it represents a tile size
                # and we need to compute offset as iv * block_size
                block_id = self._try_get_block_id_from_node(idx)
                
                if block_id is not None:
                    if self._is_singleton_block(block_id):
                        # Scalar indexing for singleton blocks (size == 1): rank-reduce.
                        offsets.append((idx_ssa, not idx_ssa.startswith("%")))
                        sizes.append(("1", True))
                    else:
                        offset_ssa = self._emit_block_offset(block_id, idx_ssa, hint="offset")
                        size_ssa = self._emit_block_extent(block_id, idx_ssa, hint="tile_extent")
                        offsets.append((offset_ssa, False))
                        size_is_static = not size_ssa.startswith("%")
                        sizes.append((size_ssa, size_is_static))
                        if size_is_static:
                            try:
                                output_dim_sizes.append(int(size_ssa))
                            except ValueError:
                                output_dim_sizes.append(None)
                        else:
                            output_dim_sizes.append(None)
                        retained_dim_positions.append(i)
                else:
                    # Generic scalar index - rank-reduce this dimension.
                    offsets.append((idx_ssa, False))
                    sizes.append(("1", True))
                    
            elif isinstance(idx, int):
                # Integer scalar index - rank-reduce this dimension.
                offsets.append((str(idx), True))
                sizes.append(("1", True))
            else:
                # Unknown type - default to 0 offset, size 1
                offsets.append(("0", True))
                sizes.append(("1", True))
        
        # Format offsets, sizes, strides - using inline literals or SSA values
        offsets_str = ", ".join(v for v, _ in offsets)
        sizes_str = ", ".join(v for v, _ in sizes)
        strides_str = ", ".join(["1"] * len(indices))
        
        # Extract dtype from memref type
        content = memref_type[memref_type.find('<')+1 : memref_type.rfind('>')]
        if 'x' in content:
            dtype_str = content.split('x')[-1]
        else:
            dtype_str = content
        
        # Build output dimensions for subview type
        output_dims = []
        for dim_size in output_dim_sizes:
            if dim_size is not None:
                output_dims.append(str(dim_size))
            else:
                output_dims.append("?")
        
        memref_dims = self._parse_plain_memref_dimensions(memref_type, "store")
        
        # Compute strides from source memref dimensions (row-major)
        # For source memref<D0xD1xD2xf32>, strides are [D1*D2, D2, 1]
        def compute_strides_from_dims(dims: list[str]) -> list[str]:
            """Compute row-major strides from dimension strings (dynamic dims -> '?')."""
            strides: list[str] = []
            running_product: int | None = 1
            for dim in reversed(dims):
                strides.append(str(running_product) if running_product is not None else "?")
                try:
                    dim_int = int(dim)
                except ValueError:
                    running_product = None
                else:
                    if running_product is not None:
                        running_product *= dim_int
            return list(reversed(strides))
        
        src_strides = compute_strides_from_dims(memref_dims)
        has_dynamic_offset = any(not is_static for _, is_static in offsets)

        if output_dims:
            if has_dynamic_offset:
                if retained_dim_positions:
                    layout_strides = [src_strides[p] for p in retained_dim_positions]
                else:
                    layout_strides = src_strides
                strides_layout_str = ", ".join(layout_strides)
                subview_type = f"memref<{'x'.join(output_dims)}x{dtype_str}, strided<[{strides_layout_str}], offset: ?>>"
            else:
                subview_type = f"memref<{'x'.join(output_dims)}x{dtype_str}>"
        else:
            if has_dynamic_offset:
                subview_type = f"memref<{dtype_str}, strided<[], offset: ?>>"
            else:
                subview_type = f"memref<{dtype_str}>"

        subview_ssa = self.mlir_output_helper.fresh("subview")
        self.mlir_output_helper.emit(
            f'{subview_ssa} = memref.subview {memref_ssa}[{offsets_str}][{sizes_str}][{strides_str}] : '
            f'{memref_type} to {subview_type}'
        )

        value_memref_ssa = self.mlir_output_helper.fresh("value_memref")
        self.mlir_output_helper.emit(
            f'{value_memref_ssa} = bufferization.to_buffer {value_ssa} : {value_type} to {subview_type}'
        )

        self.mlir_output_helper.emit(
            f'memref.copy {value_memref_ssa}, {subview_ssa} : {subview_type} to {subview_type}'
        )
        
        # Store operations don't have a useful result, but register node for downstream reference
        self.ctx.node_values[node.name] = subview_ssa
        return subview_ssa

    def visit_atomic_add(self, node: fx.Node) -> str:
        """Generate memref.subview + loom.sum for atomic accumulation.

        Requirements:
        1. The atomic_add destination must be a host tensor (memref).
        2. Lowering emits:
           - memref.subview: computes the destination tile view (load-like indexing)
           - loom.sum: accumulates the provided value into that subview
        """
        tensor_node = node.args[0]
        indices = node.args[1]
        value = node.args[2]

        if not isinstance(tensor_node, fx.Node):
            raise RuntimeError(
                f"Expected fx.Node for atomic_add target, got {type(tensor_node)}: {tensor_node}"
            )
        if not isinstance(value, fx.Node):
            raise RuntimeError(
                f"Expected fx.Node for atomic_add value, got {type(value)}: {value}"
            )

        memref_ssa = self.ctx.node_values.get(tensor_node.name, f"%{tensor_node.name}")
        memref_type = self._get_tensor_type(tensor_node)
        if not memref_type.startswith("memref<"):
            raise RuntimeError(
                f"hl.atomic_add target must be a host tensor (memref), got {memref_type} "
                f"for node {tensor_node.name}"
            )

        value_ssa = self.ctx.node_values.get(value.name, f"%{value.name}")
        value_type = self._get_tensor_type(value)

        # Build subview access pattern using the same indexing strategy as visit_load:
        # - BlockSize-origin index => offset = iv * block_size, size = block_size
        # - Scalar index => size = 1
        # - Full slice => offset = 0, size = source dim
        offsets = []
        sizes = []
        output_dim_sizes = []
        retained_dim_positions = []

        src_dims = self._parse_plain_memref_dimensions(memref_type, "atomic_add")

        for i, idx in enumerate(indices):
            if isinstance(idx, slice):
                self._validate_full_unit_slice(idx, "atomic_add", i)
                offsets.append(("0", True))

                src_dim_static = None
                if i < len(src_dims):
                    if src_dims[i] != "?":
                        try:
                            src_dim_static = int(src_dims[i])
                        except ValueError:
                            pass

                if src_dim_static is not None:
                    sizes.append((str(src_dim_static), True))
                else:
                    dim_idx_ssa = self.mlir_output_helper.fresh("dim_idx")
                    self.mlir_output_helper.emit(f"{dim_idx_ssa} = arith.constant {i} : index")
                    dim_ssa = self.mlir_output_helper.fresh("dim")
                    self.mlir_output_helper.emit(
                        f"{dim_ssa} = memref.dim {memref_ssa}, {dim_idx_ssa} : {memref_type}"
                    )
                    sizes.append((dim_ssa, False))
                output_dim_sizes.append(src_dim_static)
                retained_dim_positions.append(i)

            elif isinstance(idx, fx.Node):
                idx_ssa = self.ctx.node_values.get(idx.name, f"%{idx.name}")
                range_block_id = self.range_index_block_ids.get(idx.name)
                if range_block_id is not None:
                    size_val, size_is_static = self._get_block_size_value(range_block_id)
                    offsets.append((idx_ssa, not idx_ssa.startswith("%")))
                    sizes.append((size_val, size_is_static))
                    if size_is_static:
                        try:
                            output_dim_sizes.append(int(size_val))
                        except ValueError:
                            output_dim_sizes.append(None)
                    else:
                        output_dim_sizes.append(None)
                    retained_dim_positions.append(i)
                    continue
                block_id = self._try_get_block_id_from_node(idx)
                if block_id is not None:
                    if self._is_singleton_block(block_id):
                        offsets.append((idx_ssa, not idx_ssa.startswith("%")))
                        sizes.append(("1", True))
                    else:
                        offset_ssa = self._emit_block_offset(block_id, idx_ssa, hint="offset")
                        size_ssa = self._emit_block_extent(block_id, idx_ssa, hint="tile_extent")
                        offsets.append((offset_ssa, False))
                        size_is_static = not size_ssa.startswith("%")
                        sizes.append((size_ssa, size_is_static))
                        if size_is_static:
                            try:
                                output_dim_sizes.append(int(size_ssa))
                            except ValueError:
                                output_dim_sizes.append(None)
                        else:
                            output_dim_sizes.append(None)
                        retained_dim_positions.append(i)
                else:
                    offsets.append((idx_ssa, False))
                    sizes.append(("1", True))

            elif isinstance(idx, int):
                offsets.append((str(idx), True))
                sizes.append(("1", True))
            else:
                raise RuntimeError(f"Unsupported atomic_add index type: {type(idx)}")

        content = memref_type[memref_type.find("<") + 1 : memref_type.rfind(">")]
        if "," in content:
            content = content.split(",")[0]
        if "x" in content:
            dtype_str = content.split("x")[-1]
        else:
            dtype_str = content

        output_dims = []
        for dim_size in output_dim_sizes:
            if dim_size is not None:
                output_dims.append(str(dim_size))
            else:
                output_dims.append("?")

        def compute_strides_from_dims(dims: list[str]) -> list[str]:
            strides: list[str] = []
            running_product: int | None = 1
            for dim in reversed(dims):
                strides.append(str(running_product) if running_product is not None else "?")
                try:
                    dim_int = int(dim)
                except ValueError:
                    running_product = None
                else:
                    if running_product is not None:
                        running_product *= dim_int
            return list(reversed(strides))

        src_strides = compute_strides_from_dims(src_dims)
        has_dynamic_offset = any(not is_static for _, is_static in offsets)
        if output_dims:
            if has_dynamic_offset:
                if retained_dim_positions:
                    layout_strides = [src_strides[p] for p in retained_dim_positions]
                else:
                    layout_strides = src_strides
                strides_layout = ", ".join(layout_strides)
                subview_type = (
                    f"memref<{'x'.join(output_dims)}x{dtype_str}, "
                    f"strided<[{strides_layout}], offset: ?>>"
                )
            else:
                subview_type = f"memref<{'x'.join(output_dims)}x{dtype_str}>"
        else:
            if has_dynamic_offset:
                subview_type = f"memref<{dtype_str}, strided<[], offset: ?>>"
            else:
                subview_type = f"memref<{dtype_str}>"

        offsets_str = ", ".join(v for v, _ in offsets)
        sizes_str = ", ".join(v for v, _ in sizes)
        strides_str = ", ".join(["1"] * len(indices))

        subview_ssa = self.mlir_output_helper.fresh("subview")
        self.mlir_output_helper.emit(
            f"{subview_ssa} = memref.subview {memref_ssa}[{offsets_str}][{sizes_str}][{strides_str}] : "
            f"{memref_type} to {subview_type}"
        )

        self.mlir_output_helper.emit(
            f'"loom.sum"({subview_ssa}, {value_ssa}) : ({subview_type}, {value_type}) -> ()'
        )

        self.ctx.node_values[node.name] = subview_ssa
        return subview_ssa

    def visit_gather(self, node: fx.Node) -> str:
        """Emit a linalg-style ``loom.gather`` op for the custom gather op.

        Lowering pattern for ``gathered = gather(tile_k, src)``:

          %empty    = tensor.empty(%trip_count, d0, d1, ...) : tensor<?x*src_shape>
          %gathered = "loom.gather"(%src, %empty, %iv_block_N)
                        : (tensor<*src_shape>, tensor<?x*src_shape>, index)
                        -> tensor<?x*src_shape>

        Operand convention (mirrors linalg ins/outs/across):
          arg 0  – ins:    source tile tensor
          arg 1  – outs:   output buffer (tensor.empty with leading trip-count dim)
          arg 2  – across: loop IV of the gathered axis (runtime index)

        The op is an unregistered-dialect placeholder; pass
        ``--allow-unregistered-dialect`` to mlir-opt for validation.
        """
        tile_node = node.args[0]
        src_node = node.args[1]

        if not isinstance(tile_node, fx.Node):
            raise RuntimeError(
                f"gather: first arg (tile) must be an fx.Node, got {type(tile_node)}"
            )
        if not isinstance(src_node, fx.Node):
            raise RuntimeError(
                f"gather: second arg (src) must be an fx.Node, got {type(src_node)}"
            )

        # Recover the canonical block_id from the tile node.
        block_id = self._try_get_block_id_from_node(tile_node)
        if block_id is None:
            raise RuntimeError(
                f"gather: cannot determine block_id from tile node '{tile_node.name}'. "
                "Make sure the first argument is a Tile / block-size symnode."
            )

        # ------------------------------------------------------------------
        # Trip-count SSA for this gathered axis.
        # ------------------------------------------------------------------
        trip_count_ssa = self.ctx.reduction_trip_counts.get(block_id)
        if trip_count_ssa is None:
            import math
            info = self.ctx.env.block_sizes[block_id]
            total_extent = self.ctx.get_loop_extent(block_id)
            if isinstance(info.size, int):
                trip_val = max(1, math.ceil(total_extent / info.size))
                trip_count_ssa = self.mlir_output_helper.fresh("trip_count")
                self.mlir_output_helper.emit(
                    f"{trip_count_ssa} = arith.constant {trip_val} : index"
                )
            else:
                tile_size_ssa = self.ctx.block_size_ssa[block_id]
                extent_ssa = self.mlir_output_helper.fresh("loop_extent")
                self.mlir_output_helper.emit(
                    f"{extent_ssa} = arith.constant {total_extent} : index"
                )
                trip_count_ssa = self.mlir_output_helper.fresh("trip_count")
                self.mlir_output_helper.emit(
                    f"{trip_count_ssa} = arith.ceildivui {extent_ssa}, {tile_size_ssa} : index"
                )

        # ------------------------------------------------------------------
        # Loop IV SSA of the gathered axis  (the "across" operand).
        # ------------------------------------------------------------------
        canonical_block_id = self.ctx.resolve_block_id(block_id)
        iv_ssa = self._get_block_loop_iv(canonical_block_id)

        # ------------------------------------------------------------------
        # Source tensor SSA and type.
        # ------------------------------------------------------------------
        src_ssa = self.ctx.node_values.get(src_node.name, f"%{src_node.name}")
        src_type = self._get_tensor_type(src_node)

        # Build result type: tensor<?x*src_inner_dims>.
        if src_type.startswith("tensor<") and src_type.endswith(">"):
            inner = src_type[len("tensor<"):-1]  # e.g. "?x?xf16" or "64x64xf16"
            result_type = f"tensor<?x{inner}>"
        else:
            result_type = f"tensor<?x{src_type}>"

        # ------------------------------------------------------------------
        # Resolve per-dimension SSAs for tensor.empty(%trip_count, d0, d1, …).
        # Use FakeTensor metadata when available so that static dims
        # (e.g. from hl.specialize) appear as literals rather than tensor.dim
        # calls, keeping the empty type consistent with result_type.
        # ------------------------------------------------------------------
        src_fake_tensor = src_node.meta.get("val") if isinstance(src_node, fx.Node) else None

        empty_shape_ssas: list[str] = [trip_count_ssa]  # leading trip-count dim
        result_inner_dims: list[str] = []               # dims for result_type (after '?x')

        ndim = src_fake_tensor.ndim if src_fake_tensor is not None else 0
        for dim_idx in range(ndim):
            dim_resolved = None
            is_static = False
            if src_fake_tensor is not None:
                dim_size = src_fake_tensor.size(dim_idx)
                value_str, is_static = self.resolve_dimension(dim_size, dim_idx)
                if value_str is not None:
                    dim_resolved = value_str

            if dim_resolved is None:
                # Fallback: emit tensor.dim at runtime.
                dim_idx_ssa = self.mlir_output_helper.fresh("dim_idx")
                dim_ssa = self.mlir_output_helper.fresh("dim")
                self.mlir_output_helper.emit(
                    f"{dim_idx_ssa} = arith.constant {dim_idx} : index"
                )
                self.mlir_output_helper.emit(
                    f"{dim_ssa} = tensor.dim {src_ssa}, {dim_idx_ssa} : {src_type}"
                )
                dim_resolved = dim_ssa

            empty_shape_ssas.append(dim_resolved)
            result_inner_dims.append(dim_resolved if is_static else "?")

        # Re-derive result_type using the statically-resolved inner dims so that
        # tensor.empty's result type matches what downstream consumers expect.
        dtype_suffix = src_type[src_type.rfind("x") + 1:-1]  # e.g. "f16"
        if result_inner_dims:
            inner_dims_str = "x".join(result_inner_dims)
            result_type = f"tensor<?x{inner_dims_str}x{dtype_suffix}>"
        else:
            result_type = f"tensor<?x{dtype_suffix}>"

        # ------------------------------------------------------------------
        # Emit tensor.empty for the output buffer.
        # ------------------------------------------------------------------
        empty_shape_str = ", ".join(
            s for s in empty_shape_ssas if s.startswith("%")
        )
        empty_ssa = self.mlir_output_helper.fresh("gather_out")
        self.mlir_output_helper.emit(
            f"{empty_ssa} = tensor.empty({empty_shape_str}) : {result_type}"
        )

        # ------------------------------------------------------------------
        # Emit loom.gather in linalg-style (registered op).
        # GatherOp uses AttrSizedOperandSegments; the attribute name in this
        # MLIR version is camelCase "operandSegmentSizes" (not snake_case).
        # The four optional MeshBoundsArgs are absent (size = 0 each).
        # Segment order: ins(1), init(1), across(1), ul_x(0), ul_y(0),
        #                lr_x(0), lr_y(0)
        # ------------------------------------------------------------------
        result_ssa = self.mlir_output_helper.fresh("gathered")
        self.mlir_output_helper.emit(
            f'{result_ssa} = "loom.gather"({src_ssa}, {empty_ssa}, {iv_ssa}) '
            f'{{operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0>}} '
            f': ({src_type}, {result_type}, index) -> {result_type}'
        )

        self.ctx.node_values[node.name] = result_ssa
        self.ctx.node_types[node.name] = result_type
        # Register that dim-0 of the gathered result is the trip count, not block_size.
        self.ctx.gather_dim_overrides[node.name] = {0: trip_count_ssa}
        return result_ssa

    def visit_broadcast(self, node: fx.Node) -> str:
        """Emit a placeholder ``loom.broadcast`` op for the custom broadcast op.

        Lowering pattern for ``out = broadcast(src, dim, out_shape)``:

          %empty = tensor.empty(...) : tensor<out_shape>
          %out   = "loom.broadcast"(%src, %empty, %dim)
                     {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0>}
                     : (tensor<src_shape>, tensor<out_shape>, index)
                       -> tensor<out_shape>
        """
        if len(node.args) != 3:
            raise RuntimeError(
                f"broadcast expects 3 arguments (src, dim, out_shape), got {len(node.args)}"
            )

        src_node = node.args[0]
        dim_arg = node.args[1]
        out_shape_arg = node.args[2]

        if not isinstance(src_node, fx.Node):
            raise RuntimeError(
                f"broadcast: first arg (src) must be an fx.Node, got {type(src_node)}"
            )
        if not isinstance(out_shape_arg, (list, tuple)):
            raise RuntimeError(
                f"broadcast: third arg (out_shape) must be list/tuple, got {type(out_shape_arg)}"
            )

        src_ssa = self.ctx.node_values.get(src_node.name, f"%{src_node.name}")
        src_type = self._get_tensor_type(src_node)
        dim_ssa = self._as_index_ssa(dim_arg)

        resolved_dims: list[str] = []
        dynamic_shape_ssas: list[str] = []
        for dim_idx, dim_size in enumerate(out_shape_arg):
            if isinstance(dim_size, fx.Node):
                shape_dim_ssa = self._as_index_ssa(dim_size)
                resolved_dims.append("?")
                dynamic_shape_ssas.append(shape_dim_ssa)
                continue
            value_str, is_static = self.resolve_dimension(dim_size, dim_idx)
            if value_str is None:
                raise RuntimeError(
                    f"broadcast: unable to resolve out_shape[{dim_idx}]={dim_size}"
                )
            resolved_dims.append(value_str if is_static else "?")
            if value_str.startswith("%"):
                dynamic_shape_ssas.append(value_str)

        result_type = ""
        fake_val = node.meta.get("val")
        if fake_val is not None and hasattr(fake_val, "shape"):
            result_type = self.ctx.compute_mlir_type_from_fake_tensor(fake_val)

        if not result_type:
            src_dtype = self._get_element_type_from_node(src_node)
            if resolved_dims:
                result_type = f"tensor<{'x'.join(resolved_dims)}x{src_dtype}>"
            else:
                result_type = f"tensor<{src_dtype}>"

        empty_ssa = self.mlir_output_helper.fresh("broadcast_out")
        if dynamic_shape_ssas:
            self.mlir_output_helper.emit(
                f"{empty_ssa} = tensor.empty({', '.join(dynamic_shape_ssas)}) : {result_type}"
            )
        else:
            self.mlir_output_helper.emit(f"{empty_ssa} = tensor.empty() : {result_type}")

        result_ssa = self.mlir_output_helper.fresh("broadcasted")
        self.mlir_output_helper.emit(
            f'{result_ssa} = "loom.broadcast"({src_ssa}, {empty_ssa}, {dim_ssa}) '
            f'{{operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0>}} '
            f': ({src_type}, {result_type}, index) -> {result_type}'
        )

        self.ctx.node_values[node.name] = result_ssa
        self.ctx.node_types[node.name] = result_type
        return result_ssa

    def visit_getitem(self, node: fx.Node) -> str:
        """Map getitem to the corresponding loop result.
        
        For multi-value lowered loops, getitem(_for_loop, i) extracts
        the i-th return value as %result#i syntax.
        """
        source = node.args[0]
        index = node.args[1]
        
        # For _for_loop results, getitem extracts the i-th return value
        source_ssa = self.ctx.node_values.get(source.name, f"%{source.name}") if isinstance(source, fx.Node) else str(source)
        
        # Check if this is extracting from a multi-value loop result
        if isinstance(source, fx.Node) and source.name in self.ctx.loop_result_values:
            # Multi-value loop result - use #index syntax
            result_ssa = self.ctx.lookup_loop_result_projection(source.name, index) or f"{source_ssa}#{index}"
            self.ctx.bind_node_value(node.name, result_ssa)
            
            # Also register the tensor type from FakeTensor metadata
            if 'val' in node.meta:
                tensor_type = self.ctx.compute_mlir_type_from_fake_tensor(node.meta['val'])
                self.ctx.node_types[node.name] = tensor_type
            
            return result_ssa
        
        # For single-return loops, just use the result directly
        self.ctx.bind_node_value(node.name, source_ssa)
        return source_ssa

    
    def visit_tile_index(self, node: fx.Node) -> str:
        """Record tile_index(block_size_X) as a range-index producer."""
        block_size_node = node.args[0]
        block_id = self._try_get_block_id_from_node(block_size_node)
        if block_id is None:
            raise ValueError(f"Cannot determine block_id for tile_index node {node.name}")
        self.range_index_block_ids[node.name] = block_id
        # tile_index values are consumed by aten.add(...), which rewrites to a base index.
        placeholder = self._emit_index_constant(0)
        self.ctx.node_values[node.name] = placeholder
        self.ctx.node_types[node.name] = "index"
        return placeholder

    def visit_tile_id(self, node: fx.Node) -> str:
        """Map tile_id(block_size_X) to the corresponding parallel loop IV."""
        block_size_node = node.args[0]
        block_id = self._try_get_block_id_from_node(block_size_node)
        if block_id is None:
            raise ValueError(f"Cannot determine block_id for tile_id node {node.name}")
        bounds = self._get_active_loop_bounds(block_id)
        if bounds is None:
            iv_ssa = self._get_block_loop_iv(block_id)
            self.ctx.node_values[node.name] = iv_ssa
            return iv_ssa

        bs_ssa = self.ctx.block_size_ssa[block_id]
        start_ssa = self._emit_block_offset(block_id, bs_ssa, hint="tile_id_start")
        tile_id_ssa = self._emit_index_divui(start_ssa, bs_ssa, hint="tile_id")
        self.ctx.node_values[node.name] = tile_id_ssa
        return tile_id_ssa

    def visit_tile_begin(self, node: fx.Node) -> str:
        """Map tile_begin(block_size_X) to lb + IV * block_size when active."""
        block_size_node = node.args[0]
        block_id = self._try_get_block_id_from_node(block_size_node)
        if block_id is None:
            raise ValueError(f"Cannot determine block_id for tile_begin node {node.name}")
        canonical_id = self.ctx.resolve_block_id(block_id)
        bs_ssa = self.ctx.block_size_ssa[canonical_id]
        result = self._emit_block_offset(canonical_id, bs_ssa, hint="tile_begin")
        self.ctx.node_values[node.name] = result
        return result

    def visit_tile_end(self, node: fx.Node) -> str:
        """Map tile_end(block_size_X) to min(lb + (IV + 1) * block_size, ub) when active."""
        block_size_node = node.args[0]
        block_id = self._try_get_block_id_from_node(block_size_node)
        if block_id is None:
            raise ValueError(f"Cannot determine block_id for tile_end node {node.name}")
        canonical_id = self.ctx.resolve_block_id(block_id)
        bs_ssa = self.ctx.block_size_ssa[canonical_id]
        result = self._emit_block_end(canonical_id, bs_ssa, hint="tile_end")
        self.ctx.node_values[node.name] = result
        return result

    def visit_comparison(self, node: fx.Node, predicate: str) -> str:
        """Emit arith.cmpi for integer/index comparisons (eq, ne, slt, etc.)."""
        lhs_node = node.args[0]
        rhs_val = node.args[1]

        # Resolve LHS
        if isinstance(lhs_node, fx.Node):
            lhs_ssa = self.ctx.node_values.get(lhs_node.name, f"%{lhs_node.name}")
        else:
            lhs_ssa = self.mlir_output_helper.fresh("cmp_lhs")
            self.mlir_output_helper.emit(
                f'{lhs_ssa} = arith.constant {int(lhs_node)} : index'
            )

        # Resolve RHS
        if isinstance(rhs_val, fx.Node):
            rhs_ssa = self.ctx.node_values.get(rhs_val.name, f"%{rhs_val.name}")
        elif isinstance(rhs_val, (int, float)):
            rhs_ssa = self.mlir_output_helper.fresh("cmp_rhs")
            self.mlir_output_helper.emit(
                f'{rhs_ssa} = arith.constant {int(rhs_val)} : index'
            )
        else:
            raise RuntimeError(f"Unsupported comparison RHS: {rhs_val}")

        result = self.mlir_output_helper.fresh("cmp")
        self.mlir_output_helper.emit(
            f'{result} = arith.cmpi {predicate}, {lhs_ssa}, {rhs_ssa} : index'
        )
        self.ctx.node_values[node.name] = result
        return result

    def visit_if(self, node: fx.Node) -> str | None:
        """Generate scf.if and visit the IfGraphInfo body."""
        test_node = node.args[0]       # The i1 condition
        graph_id = node.args[1]        # IfGraphInfo graph_id
        args = node.args[2]            # Values passed to the if-body

        # Resolve condition SSA
        if isinstance(test_node, fx.Node):
            cond_ssa = self.ctx.node_values.get(test_node.name, f"%{test_node.name}")
        else:
            raise RuntimeError(f"Unsupported _if condition: {test_node}")

        # Get the IfGraphInfo
        if_graph = self.ctx.graphs.get(graph_id)
        if if_graph is None:
            raise ValueError(f"IfGraphInfo with graph_id={graph_id} not registered")

        # Emit scf.if (no results for side-effect-only body)
        self.mlir_output_helper.emit(f'scf.if {cond_ssa} {{')
        self.mlir_output_helper.push()

        # Map if-body placeholders to outer-scope SSA values
        old_loop_iter_args = self.loop_iter_args.copy()
        for i, a in enumerate(args):
            placeholder_name = f"arg{i}_1"
            if isinstance(a, fx.Node):
                self._propagate_metadata(a.name, placeholder_name, is_placeholder=True)

        # Visit inner graph
        self.visit_graph(if_graph)

        # Restore
        self.loop_iter_args = old_loop_iter_args

        self.mlir_output_helper.pop()
        self.mlir_output_helper.emit("}")

        self.ctx.node_values[node.name] = None
        return None

    def visit_mask_to(self, node: fx.Node) -> str:
        """Shortcircuit _mask_to by passing through the input tensor.
        
        _mask_to is used for boundary checks in Helion. For now, we simply
        pass through the input tensor unchanged. Proper masking support
        would require implementing actual boundary logic.
        
        TODO: Implement proper masking/boundary check logic.
        """
        tensor_node = node.args[0]
        # mask_value = node.args[1]  # The fill value for out-of-bounds (ignored)
        
        # Just pass through the input tensor
        if isinstance(tensor_node, fx.Node):
            self._propagate_metadata(tensor_node.name, node.name)
            ssa = self.ctx.node_values[node.name]
        else:
            ssa = str(tensor_node)
            self.ctx.node_values[node.name] = ssa
        
        return ssa
    
    def visit_subscript(self, node: fx.Node) -> str:
        """Generate tensor ops for subscript (indexing/slicing/newaxis).
        
        Replaces helion.subscript with tensor.extract_slice and tensor.expand_shape.
        
        1. tensor.extract_slice: Used for slices (:) and integer indexing (rank reducing).
        2. tensor.expand_shape: Used for adding new dimensions (None).
        """
        tensor_node = node.args[0]
        indices = node.args[1] if len(node.args) > 1 else []
        
        if not isinstance(tensor_node, fx.Node):
             raise RuntimeError(f"Expected fx.Node for tensor_node, got {type(tensor_node)}: {tensor_node}")
        current_ssa = self.ctx.node_values.get(tensor_node.name, f"%{tensor_node.name}")
        source_type = self._get_tensor_type(tensor_node)
        
        # -----------------------------------------------------------
        # Step 1: Handle Slicing and Integer Indexing (tensor.extract_slice)
        # -----------------------------------------------------------
        
        # Filter out None (newaxis) to check for slicing requirements
        slice_indices = [idx for idx in indices if idx is not None]
        source_fake_tensor = tensor_node.meta.get("val") if isinstance(tensor_node, fx.Node) else None
        
        needs_slicing = False
        # Default behavior: slice when explicit indices are provided.
        if slice_indices:
            needs_slicing = True

        # Optimization: skip identity extract_slice for pure full-range slicing
        # (e.g. t[:, :, None]) when rank is unchanged by indexing.
        if slice_indices and source_fake_tensor is not None:
            has_non_slice_index = any(isinstance(idx, (int, fx.Node)) for idx in slice_indices)
            is_full_range_slice = (
                isinstance(idx, slice)
                and idx.start is None
                and idx.stop is None
                and idx.step is None
                for idx in slice_indices
            )
            all_full_range_slices = all(is_full_range_slice)
            rank_preserving_slice = len(slice_indices) == source_fake_tensor.ndim
            if all_full_range_slices and (not has_non_slice_index) and rank_preserving_slice:
                needs_slicing = False
        
        extracted_ssa = current_ssa
        # Default intermediate type is source type
        extracted_type = source_type
        
        source_dim_for_extracted = [] # Track which source dims were preserved
        
        if needs_slicing:
            offsets_ssa = []
            sizes_ssa = []
            strides_ssa = []
            
            input_dim_idx = 0
            
            for idx in slice_indices:
                if isinstance(idx, slice):
                    self._validate_full_unit_slice(idx, "subscript", input_dim_idx)
                    # Handle slice(None) aka [:]
                    # Offset 0 - use inline literal
                    offsets_ssa.append("0")
                    
                    # Size: try to resolve from FakeTensor
                    dim_resolved = None
                    if source_fake_tensor is not None and input_dim_idx < source_fake_tensor.ndim:
                        dim_size = source_fake_tensor.size(input_dim_idx)
                        # Pass gather_dim_overrides for the source tensor
                        overrides = self.ctx.gather_dim_overrides.get(tensor_node.name)
                        value_str, is_static = self.resolve_dimension(dim_size, input_dim_idx, overrides=overrides)
                        if value_str is not None:
                            dim_resolved = value_str
                    
                    if dim_resolved is None:
                        # Fallback to tensor.dim
                        dim_ssa = self.mlir_output_helper.fresh("dim")
                        dim_idx_ssa = self.mlir_output_helper.fresh("dim_idx")
                        self.mlir_output_helper.emit(f'{dim_idx_ssa} = arith.constant {input_dim_idx} : index')
                        self.mlir_output_helper.emit(f'{dim_ssa} = tensor.dim {current_ssa}, {dim_idx_ssa} : {source_type}')
                        dim_resolved = dim_ssa
                    
                    sizes_ssa.append(dim_resolved)
                    strides_ssa.append("1")  # Default stride 1
                    source_dim_for_extracted.append(input_dim_idx)
                    input_dim_idx += 1
                    
                elif isinstance(idx, int) or isinstance(idx, fx.Node):
                    # Integer indexing: offset=idx, size=1, stride=1
                    # Rank reducing -> we will drop this dimension in the result type
                    
                    if isinstance(idx, fx.Node):
                        idx_val = self.ctx.node_values.get(idx.name, f"%{idx.name}")
                    else:
                        # Use inline literal for constant integers
                        idx_val = str(idx)
                        
                    offsets_ssa.append(idx_val)
                    sizes_ssa.append("1")  # Inline literal
                    strides_ssa.append("1")
                    
                    input_dim_idx += 1
            
            # Build result dims: static literal when the size is known at compile
            # time (resolve_dimension returned a non-SSA value), '?' otherwise.
            result_dims = []
            sizes_iter = iter(sizes_ssa)
            for idx in slice_indices:
                size_val = next(sizes_iter)
                if isinstance(idx, slice):
                    # Keep this dimension; use the literal when static.
                    result_dims.append(size_val if not size_val.startswith("%") else "?")
                    
            # Derive dtype from source FakeTensor
            source_dtype = self._get_element_type_from_node(tensor_node)
            dims_str = "x".join(result_dims)
            result_type = f"tensor<{dims_str}x{source_dtype}>"
            extracted_type = result_type
            
            result_slice = self.mlir_output_helper.fresh("slice")
            
            offsets_str = ", ".join(offsets_ssa)
            sizes_str = ", ".join(sizes_ssa)
            strides_str = ", ".join(strides_ssa)
            
            self.mlir_output_helper.emit(
                f'{result_slice} = tensor.extract_slice {current_ssa}[{offsets_str}][{sizes_str}][{strides_str}] : '
                f'{source_type} to {result_type}'
            )
            
            extracted_ssa = result_slice
        else:
             # If no slicing, we still need to track preserved dims (all of them)
             if source_fake_tensor is not None:
                  source_dim_for_extracted = list(range(source_fake_tensor.ndim))

        # -----------------------------------------------------------
        # Step 2: Handle New Axis (tensor.expand_shape)
        # -----------------------------------------------------------
        
        has_newaxis = any(idx is None for idx in indices)
        
        if has_newaxis:
            output_dim_types = [] 
            for idx in indices:
                if isinstance(idx, slice):
                    output_dim_types.append('real')
                elif idx is None:
                    output_dim_types.append('new')
            
            input_dim_assignments = {} # int -> list[int]
            real_dim_counter = 0
            for i, dtype in enumerate(output_dim_types):
                if dtype == 'real':
                    input_dim_assignments[real_dim_counter] = [i]
                    real_dim_counter += 1
            
            for i, dtype in enumerate(output_dim_types):
                if dtype == 'new':
                    found_next = False
                    target_input_dim = -1
                    for k in range(real_dim_counter):
                        if input_dim_assignments[k][0] > i:
                            target_input_dim = k
                            break
                    if target_input_dim != -1:
                        input_dim_assignments[target_input_dim].insert(0, i)
                        found_next = True
                    if not found_next:
                        last_input = real_dim_counter - 1
                        if last_input >= 0:
                            input_dim_assignments[last_input].append(i)
            
            reassoc_list = []
            for k in range(real_dim_counter):
                reassoc_list.append(input_dim_assignments[k])
            reassoc_str = "[" + ", ".join(["[" + ", ".join(map(str, grp)) + "]" for grp in reassoc_list]) + "]"
            
            source_dtype = self._get_element_type_from_node(tensor_node)
            result_expand = self.mlir_output_helper.fresh("expand")
            output_shape_ssas = []
            result_dims = []
            extracted_dim_idx = 0 

            for i, dtype in enumerate(output_dim_types):
                if dtype == 'new':
                    output_shape_ssas.append("1")
                    result_dims.append("1")
                else:
                    dim_resolved = None
                    is_static = False
                    if source_fake_tensor is not None and extracted_dim_idx < len(source_dim_for_extracted):
                        source_dim = source_dim_for_extracted[extracted_dim_idx]
                        if source_dim < source_fake_tensor.ndim:
                            dim_size = source_fake_tensor.size(source_dim)
                            overrides = self.ctx.gather_dim_overrides.get(tensor_node.name)
                            value_str, is_static = self.resolve_dimension(dim_size, source_dim, overrides=overrides)
                            if value_str is not None:
                                dim_resolved = value_str

                    if dim_resolved is None:
                        dim_idx_ssa = self.mlir_output_helper.fresh("dim_idx")
                        dim_ssa = self.mlir_output_helper.fresh("dim")
                        self.mlir_output_helper.emit(f'{dim_idx_ssa} = arith.constant {extracted_dim_idx} : index')
                        self.mlir_output_helper.emit(f'{dim_ssa} = tensor.dim {extracted_ssa}, {dim_idx_ssa} : {extracted_type}')
                        dim_resolved = dim_ssa
                        result_dims.append("?")
                    else:
                        result_dims.append(dim_resolved if is_static else "?")

                    output_shape_ssas.append(dim_resolved)
                    extracted_dim_idx += 1

            dims_str = "x".join(result_dims)
            result_type = f"tensor<{dims_str}x{source_dtype}>"
            
            # Propagate gather_dim_overrides
            source_overrides = self.ctx.gather_dim_overrides.get(tensor_node.name)
            if source_overrides:
                result_overrides = {}
                real_idx = 0
                for i, otype in enumerate(output_dim_types):
                    if otype == 'real':
                        src_dim = source_dim_for_extracted[real_idx]
                        if src_dim in source_overrides:
                            result_overrides[i] = source_overrides[src_dim]
                        real_idx += 1
                if result_overrides:
                    self.ctx.gather_dim_overrides[node.name] = result_overrides

            self.mlir_output_helper.emit(f'{result_expand} = tensor.expand_shape {extracted_ssa} {reassoc_str} output_shape [{", ".join(output_shape_ssas)}] : {extracted_type} into {result_type}')
            self.ctx.node_values[node.name] = result_expand
            self.ctx.node_types[node.name] = result_type
            return result_expand
        
        self.ctx.node_values[node.name] = extracted_ssa
        self.ctx.node_types[node.name] = extracted_type
        
        # Propagate overrides for Step 1 result
        source_overrides = self.ctx.gather_dim_overrides.get(tensor_node.name)
        if source_overrides:
            result_overrides = {}
            for i, src_dim in enumerate(source_dim_for_extracted):
                 if src_dim in source_overrides:
                      result_overrides[i] = source_overrides[src_dim]
            if result_overrides:
                 self.ctx.gather_dim_overrides[node.name] = result_overrides
        return extracted_ssa
    
    def visit_aten_compute(self, node: fx.Node) -> str:
        """Generate MLIR for ATen compute ops using torch-mlir.
        
        Uses torch-mlir's FxImporter to generate MLIR for ATen operations.
        The output is always lowered to linalg-on-tensors.
        
        Optimizes tensor.dim operations by passing pre-existing dimension SSAs.
        """
        from helion._compiler.variable_origin import BlockSizeOrigin
        target = node.target

        # Fast path: lower broadcast-compatible repeat directly to linalg.broadcast.
        repeat_result = self._try_lower_repeat_as_linalg_broadcast(node)
        if repeat_result is not None:
            return repeat_result
        
        # Resolve all operands to SSA values
        operand_ssas = []
        for arg in node.args:
            if isinstance(arg, fx.Node):
                operand_ssas.append(self.ctx.node_values.get(arg.name, f"%{arg.name}"))
            elif arg is not None:
                operand_ssas.append(str(arg))
        
        # -------------------------------------------------------------------------
        # Use torch-mlir to generate MLIR for all ATen operations
        # The output is always linalg-on-tensors
        # -------------------------------------------------------------------------
        
        # Collect SSA values for tensor operands (matching what import_aten_node_to_mlir expects as args)
        tensor_operands = []
        tensor_operand_nodes = []  # Keep track of the actual nodes for dimension analysis
        def collect_operands(arg):
            if isinstance(arg, fx.Node):
                tensor_operands.append(self.ctx.node_values.get(arg.name, f"%{arg.name}"))
                tensor_operand_nodes.append(arg)
            return arg
            
        fx.map_arg(node.args, collect_operands)
        fx.map_arg(node.kwargs, collect_operands)

        precise_import_inputs: list[object] = []

        def collect_precise_import_inputs(arg):
            if isinstance(arg, fx.Node):
                precise_import_inputs.append(self._create_precise_import_value(arg))
            return arg

        fx.map_arg(node.args, collect_precise_import_inputs)
        fx.map_arg(node.kwargs, collect_precise_import_inputs)

        # Use torch-mlir to generate MLIR for this operation
        mlir_text = import_aten_node_to_mlir(
            node,
            input_tensors=precise_import_inputs,
        )
        if mlir_text is None:
             raise RuntimeError(f"Failed to lower ATen op: {node.name} ({target})")
        
        # -------------------------------------------------------------------------
        # Build dimension_ssa_map for tensor.dim optimization
        # Maps each tensor operand SSA to a list of dimension SSAs
        # -------------------------------------------------------------------------
        host_function = self.ctx.bound_kernel.host_function
        shape_env = self.ctx.bound_kernel.env.shape_env
        
        dimension_ssa_map = {}
        
        for operand_ssa, operand_node in zip(tensor_operands, tensor_operand_nodes):
            fake_tensor = operand_node.meta.get("val") if operand_node else None
            if fake_tensor is None or not hasattr(fake_tensor, 'ndim'):
                continue
                
            dim_ssas = []
            for dim_idx in range(fake_tensor.ndim):
                dim_size = fake_tensor.size(dim_idx)
                
                if hasattr(dim_size, '_sympy_'):
                    # It's a SymInt - look up its origin
                    sym = dim_size._sympy_()
                    origin_info = host_function.expr_to_origin.get(sym)
                    origin = origin_info.origin if origin_info else None
                    
                    if isinstance(origin, BlockSizeOrigin):
                        raw_id = origin.block_id
                        canonical_id = self.ctx.resolve_block_id(raw_id)
                        block_info = self.ctx.env.block_sizes[raw_id]
                        
                        if isinstance(block_info.size, int):
                            dim_ssas.append(str(block_info.size))
                        else:
                            overrides = self.ctx.gather_dim_overrides.get(operand_node.name)
                            if overrides and dim_idx in overrides:
                                dim_ssas.append(overrides[dim_idx])
                            else:
                                ssa = self.ctx.block_size_ssa.get(canonical_id)
                                if ssa:
                                    dim_ssas.append(ssa)
                                else:
                                    dim_ssas.append(None)
                    else:
                        if sym in shape_env.var_to_val:
                            concrete_val = int(shape_env.var_to_val[sym])
                            dim_ssas.append(str(concrete_val))
                        else:
                            dim_ssas.append(None)
                else:
                    # Concrete int dimension
                    dim_ssas.append(str(int(dim_size)))
            
            if any(d is not None for d in dim_ssas):
                dimension_ssa_map[operand_ssa] = dim_ssas
        
        # Identify scalar (non-tensor) operands so inline_torch_mlir_output
        # can capture them directly in linalg.generic body instead of as
        # ins() tensor operands.
        scalar_operand_map = {}
        for i, op_node in enumerate(tensor_operand_nodes):
            if isinstance(op_node, fx.Node):
                node_type = self.ctx.node_types.get(op_node.name, "")
                if node_type and not node_type.startswith("tensor<") and not node_type.startswith("memref<"):
                    scalar_operand_map[i] = tensor_operands[i]

        from .torch_mlir_helper import inline_torch_mlir_output
        result = inline_torch_mlir_output(
            mlir_text,
            tensor_operands,
            self.mlir_output_helper,
            dimension_ssa_map=dimension_ssa_map if dimension_ssa_map else None,
            scalar_operand_map=scalar_operand_map if scalar_operand_map else None,
        )
        
        self.ctx.node_values[node.name] = result
        imported_result_type = self._extract_imported_result_type(mlir_text)
        if "val" in node.meta:
            val = node.meta["val"]
            if imported_result_type is not None:
                self.ctx.node_types[node.name] = imported_result_type
            elif hasattr(val, "shape"):
                self.ctx.node_types[node.name] = self.ctx.compute_mlir_type_from_fake_tensor(val)
            elif hasattr(val, "dtype"):
                self.ctx.node_types[node.name] = torch_dtype_to_mlir_element_type(val.dtype)
            
            # Propagate gather_dim_overrides from inputs to outputs
            if hasattr(val, 'shape'):
                output_overrides = {}
                for dim_idx, dim_size in enumerate(val.shape):
                    if hasattr(dim_size, '_sympy_'):
                        sym = dim_size._sympy_()
                        # Check if any input operand has an override for this symbol
                        for op_node in tensor_operand_nodes:
                            if not isinstance(op_node, fx.Node):
                                continue
                            op_val = op_node.meta.get("val")
                            if op_val is None or not hasattr(op_val, 'shape'):
                                continue
                            for inp_dim, inp_size in enumerate(op_val.shape):
                                if hasattr(inp_size, '_sympy_') and inp_size._sympy_() == sym:
                                    overrides = self.ctx.gather_dim_overrides.get(op_node.name)
                                    if overrides and inp_dim in overrides:
                                        output_overrides[dim_idx] = overrides[inp_dim]
                                        break
                            if dim_idx in output_overrides:
                                break
                if output_overrides:
                    self.ctx.gather_dim_overrides[node.name] = output_overrides
                    
        return result

    def _try_lower_repeat_as_linalg_broadcast(self, node: fx.Node) -> str | None:
        """Lower aten.repeat.default directly to linalg.broadcast when valid.

        This is intentionally conservative and only handles rank-preserving
        repeat where each repeated dimension is either:
        - repeat factor 1, or
        - source extent statically 1 (broadcastable expansion).
        """
        target = node.target
        if str(target) != "aten.repeat.default":
            return None

        if len(node.args) < 2:
            return None

        tensor_node = node.args[0]
        repeats = node.args[1]
        if not isinstance(tensor_node, fx.Node):
            return None
        if not isinstance(repeats, (list, tuple)):
            return None
        if not all(isinstance(r, int) and r >= 1 for r in repeats):
            return None

        src_val = tensor_node.meta.get("val")
        out_val = node.meta.get("val")
        if src_val is None or out_val is None:
            return None
        if not hasattr(src_val, "shape") or not hasattr(out_val, "shape"):
            return None

        src_rank = len(src_val.shape)
        out_rank = len(out_val.shape)
        if src_rank != out_rank or len(repeats) != src_rank:
            return None

        expanded_dims = [i for i, rep in enumerate(repeats) if rep != 1]
        if not expanded_dims:
            # No-op repeat.
            input_ssa = self.ctx.node_values.get(tensor_node.name, f"%{tensor_node.name}")
            input_type = self._get_tensor_type(tensor_node)
            self.ctx.node_values[node.name] = input_ssa
            self.ctx.node_types[node.name] = input_type
            return input_ssa

        # Guard: all expanded dims must come from static-one dimensions.
        for dim_idx in expanded_dims:
            src_dim = src_val.shape[dim_idx]
            if not (isinstance(src_dim, int) and src_dim == 1):
                return None

        # Conservative shape: only support suffix expansion dims (common case:
        # keepdim=True then repeat on trailing axis).
        expanded_start = expanded_dims[0]
        if expanded_dims != list(range(expanded_start, src_rank)):
            return None
        if expanded_start == 0:
            return None

        input_ssa = self.ctx.node_values.get(tensor_node.name, f"%{tensor_node.name}")
        input_type = self._get_tensor_type(tensor_node)
        source_dtype = self._get_element_type_from_node(tensor_node)

        # Collapse trailing singleton expanded dims into the preceding kept dim.
        collapse_reassoc_groups: list[list[int]] = []
        for i in range(expanded_start - 1):
            collapse_reassoc_groups.append([i])
        collapse_reassoc_groups.append(list(range(expanded_start - 1, src_rank)))

        collapsed_dim_tokens: list[str] = []
        src_overrides = self.ctx.gather_dim_overrides.get(tensor_node.name, {})
        for i in range(expanded_start):
            dim_size = src_val.shape[i]
            value_str, is_static = self.resolve_dimension(
                dim_size,
                i,
                overrides=src_overrides if src_overrides else None,
            )
            if value_str is None:
                collapsed_dim_tokens.append("?")
            else:
                collapsed_dim_tokens.append(value_str if is_static else "?")

        collapsed_dims_str = "x".join(collapsed_dim_tokens)
        collapsed_type = f"tensor<{collapsed_dims_str}x{source_dtype}>"
        collapse_reassoc_str = "[" + ", ".join(
            "[" + ", ".join(str(v) for v in grp) + "]" for grp in collapse_reassoc_groups
        ) + "]"
        collapsed_ssa = self.mlir_output_helper.fresh("collapse")
        self.mlir_output_helper.emit(
            f"{collapsed_ssa} = tensor.collapse_shape {input_ssa} {collapse_reassoc_str} : "
            f"{input_type} into {collapsed_type}"
        )

        # Build output tensor type and dynamic shape operands.
        output_shape_ssas: list[str] = []
        output_dim_tokens: list[str] = []

        # Preserve dynamic dim overrides for dimensions unchanged by repeat=1.
        out_overrides: dict[int, str] = {}

        for dim_idx, rep in enumerate(repeats):
            src_dim = src_val.shape[dim_idx]
            if rep == 1:
                value_str, is_static = self.resolve_dimension(
                    src_dim,
                    dim_idx,
                    overrides=src_overrides if src_overrides else None,
                )
                if value_str is None:
                    dim_idx_ssa = self.mlir_output_helper.fresh("dim_idx")
                    dim_ssa = self.mlir_output_helper.fresh("dim")
                    self.mlir_output_helper.emit(f"{dim_idx_ssa} = arith.constant {dim_idx} : index")
                    self.mlir_output_helper.emit(f"{dim_ssa} = tensor.dim {input_ssa}, {dim_idx_ssa} : {input_type}")
                    output_shape_ssas.append(dim_ssa)
                    output_dim_tokens.append("?")
                else:
                    output_shape_ssas.append(value_str)
                    output_dim_tokens.append(value_str if is_static else "?")
                    if dim_idx in src_overrides:
                        out_overrides[dim_idx] = src_overrides[dim_idx]
            else:
                # rep > 1 and source extent is statically 1: output extent is rep.
                output_shape_ssas.append(str(rep))
                output_dim_tokens.append(str(rep))

        output_dims_str = "x".join(output_dim_tokens)
        output_type = f"tensor<{output_dims_str}x{source_dtype}>"

        dynamic_shape_ssas = [s for s in output_shape_ssas if s.startswith("%")]
        empty_ssa = self.mlir_output_helper.fresh("empty")
        if dynamic_shape_ssas:
            self.mlir_output_helper.emit(
                f"{empty_ssa} = tensor.empty({', '.join(dynamic_shape_ssas)}) : {output_type}"
            )
        else:
            self.mlir_output_helper.emit(f"{empty_ssa} = tensor.empty() : {output_type}")

        dims_attr = ", ".join(str(i) for i in expanded_dims)
        result_ssa = self.mlir_output_helper.fresh("broadcast")
        self.mlir_output_helper.emit(
            f"{result_ssa} = linalg.broadcast ins({collapsed_ssa} : {collapsed_type}) "
            f"outs({empty_ssa} : {output_type}) dimensions = [{dims_attr}]"
        )

        self.ctx.node_values[node.name] = result_ssa
        self.ctx.node_types[node.name] = output_type
        if out_overrides:
            self.ctx.gather_dim_overrides[node.name] = out_overrides
        return result_ssa

    def _get_element_type_from_node(self, node: fx.Node) -> str:
        """Extract MLIR element type string from a node's FakeTensor dtype."""
        from .mlir_utils import torch_dtype_to_mlir_element_type
        if isinstance(node, fx.Node):
            fake = node.meta.get("val")
            if fake is not None and hasattr(fake, "dtype"):
                return self.ctx.type_resolver.get_element_type_from_node(node)
        return "f32"  # Last-resort fallback
    
    def _get_tensor_type(self, tensor_node: fx.Node | str) -> str:
        """Get MLIR tensor type for a tensor node.
        
        Lookup order:
        1. ctx.node_types (already computed for this node)
        2. ctx.host_tensor_types (pre-computed for _host_tensor nodes)
        3. ctx.arg_mlir_types (kernel arguments)
        4. Compute from FakeTensor if available
        """
        if isinstance(tensor_node, fx.Node):
            name = tensor_node.name
            node = tensor_node
        else:
            name = str(tensor_node)
            node = None
        
        # Look up in registered node types
        if name in self.ctx.node_types:
            return self.ctx.node_types[name]
        
        # Look up in host tensor types (pre-computed in LoweringContext)
        if name in self.ctx.host_tensor_types:
            return self.ctx.host_tensor_types[name]
            
        # Look up in arg_mlir_types
        if name in self.ctx.arg_mlir_types:
            return self.ctx.arg_mlir_types[name]
        
        # Try to compute from FakeTensor in node metadata
        if node is not None and 'val' in node.meta:
            fake_tensor = node.meta['val']
            return self.ctx.type_resolver.compute_mlir_type_from_fake_tensor(fake_tensor)
        
        raise RuntimeError(f"Cannot compute MLIR type for node {name}")

    def _create_precise_import_value(self, node: fx.Node) -> object:
        """Create a more precise fake value for torch-mlir import.

        Dims that shape_env proves static are materialized as concrete ints.
        Dims that still depend on block sizes remain symbolic.
        """
        from torch._subclasses.fake_tensor import FakeTensorMode

        val = node.meta.get("val")

        if isinstance(val, torch.Tensor):
            resolved_shape: list[object] = []
            overrides = self.ctx.gather_dim_overrides.get(node.name)
            for dim_idx, dim_size in enumerate(val.shape):
                if hasattr(dim_size, "_sympy_"):
                    dim_value, is_static = self.resolve_dimension(dim_size, dim_idx, overrides=overrides)
                    if dim_value is not None and is_static:
                        resolved_shape.append(int(dim_value))
                    else:
                        resolved_shape.append(dim_size)
                else:
                    resolved_shape.append(int(dim_size))

            with FakeTensorMode():
                return torch.empty(resolved_shape, dtype=val.dtype, device="meta")

        if isinstance(val, (tuple, list)):
            converted = [
                self._create_precise_import_value_from_meta(elem)
                for elem in val
            ]
            return tuple(converted) if isinstance(val, tuple) else converted

        if isinstance(val, (int, float, bool)):
            return val

        raise RuntimeError(f"Unsupported import value for node {node.name}: {type(val)}")

    def _create_precise_import_value_from_meta(self, val: object) -> object:
        """Recursively clone tensor metadata into precise fake tensors."""
        from torch._subclasses.fake_tensor import FakeTensorMode

        if isinstance(val, torch.Tensor):
            resolved_shape: list[object] = []
            for dim in val.shape:
                if hasattr(dim, "_sympy_"):
                    resolved_shape.append(dim)
                else:
                    resolved_shape.append(int(dim))
            with FakeTensorMode():
                return torch.empty(resolved_shape, dtype=val.dtype, device="meta")
        if isinstance(val, (tuple, list)):
            converted = [self._create_precise_import_value_from_meta(elem) for elem in val]
            return tuple(converted) if isinstance(val, tuple) else converted
        return val

    @staticmethod
    def _extract_imported_result_type(mlir_text: str) -> str | None:
        """Extract the single return type from an imported torch-mlir function."""
        import re

        for line in mlir_text.splitlines():
            if "func.func @aten_op" not in line:
                continue
            match = re.search(r'\)\s*->\s*(.+?)\s*\{', line)
            if match:
                result_type = match.group(1).strip()
                if result_type.startswith("("):
                    return None
                return result_type
        return None
    
    def _get_hex_constant(self, value: float, dtype_str: str) -> str:
        """Get the hexadecimal representation of inf/nan for a given dtype.
        
        Args:
            value: The float value (must be inf or nan)
            dtype_str: The MLIR dtype string (f32, f16, bf16)
            
        Returns:
            Hexadecimal string representation
        """
        import math
        
        if dtype_str == "f32":
            if math.isinf(value):
                return "0xFF800000" if value < 0 else "0x7F800000"
            else:
                return "0x7FC00000"
        elif dtype_str == "f16":
            if math.isinf(value):
                return "0xFC00" if value < 0 else "0x7C00"
            else:
                return "0x7E00"
        elif dtype_str == "bf16":
            if math.isinf(value):
                return "0xFF80" if value < 0 else "0x7F80"
            else:
                return "0x7FC0"
        else:
            # Fallback to f32 for unknown types, but warn?
            # For now, just return what it would be for f32
            if math.isinf(value):
                return "0xFF800000" if value < 0 else "0x7F800000"
            else:
                return "0x7FC00000"
