"""IR Visitor for walking Device IR graphs and generating MLIR.

This module implements a visitor pattern for converting Helion Device IR
to MLIR by walking FX graph nodes instruction-by-instruction.

The visitor dispatches to specific handlers based on the node's target:
- _get_symnode -> SSA lookup via Origin (BlockSizeOrigin -> block_size_ssa)
- full -> tensor.empty + linalg.fill
- _for_loop -> affine.for + recursive visit
- _phi -> Loop result SSA (simplified merge pattern detection)
- _host_tensor -> function argument mapping
- aten.sym_size.int -> inline concrete value or tensor.dim
- load -> memref.subview + bufferization.to_tensor
- store -> bufferization.to_memref + memref.copy
- aten.* compute -> linalg-on-tensors (via torch-mlir)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.fx as fx
from torch.ops import aten

import helion.language.memory_ops as hl_memory_ops
import helion.language._tracing_ops as hl_tracing_ops
import helion.language.creation_ops as hl_creation_ops
import helion.language.view_ops as hl_view_ops
import helion.language.tile_ops as hl_tile_ops

from .mlir_utils import (
    format_attr_dict,
    format_string_attr,
    torch_dtype_to_mlir_element_type,
    format_tensor_type,
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
    
    def resolve_dimension(self, dim_size, dim_hint: int = 0) -> tuple[str, bool]:
        """Resolve a dimension size to an SSA value or inline literal.

        Block IDs are resolved through the alias map so that dimensions
        coming from the same source across grids share the same SSA.
        """
        from helion._compiler.variable_origin import BlockSizeOrigin

        if not hasattr(dim_size, '_sympy_'):
            return (str(int(dim_size)), True)

        sym = dim_size._sympy_()
        host_function = self.ctx.bound_kernel.host_function
        shape_env = self.ctx.bound_kernel.env.shape_env

        origin_info = host_function.expr_to_origin.get(sym)
        origin = origin_info.origin if origin_info else None

        if isinstance(origin, BlockSizeOrigin):
            raw_id = origin.block_id
            canonical_id = self.ctx.resolve_block_id(raw_id)
            block_info = self.ctx.env.block_sizes[raw_id]

            if isinstance(block_info.size, int):
                return (str(block_info.size), True)

            ssa = self.ctx.block_size_ssa.get(canonical_id)
            if ssa:
                return (ssa, False)

        if sym in shape_env.var_to_val:
            concrete_val = int(shape_env.var_to_val[sym])
            return (str(concrete_val), True)

        return (None, False)
        
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
            self.ctx.node_values[node.name] = ssa
            return ssa
        
        # Otherwise it's a function argument placeholder
        # For now, just create a placeholder SSA value
        ssa = f"%{node.name}"
        self.ctx.node_values[node.name] = ssa
        
        # Register type from arg_mlir_types if available
        if node.name in self.ctx.arg_mlir_types:
            self.ctx.node_types[node.name] = self.ctx.arg_mlir_types[node.name]
                
        return ssa
    
    def visit_call_function(self, node: fx.Node) -> str | None:
        """Dispatch call_function nodes to specific handlers."""
        target = node.target
        
        # _get_symnode -> loom.get_symbol
        if target is hl_tracing_ops._get_symnode:
            return self.visit_get_symnode(node)
        
        # full -> tensor.empty + linalg.fill
        if target is hl_creation_ops.full:
            return self.visit_full(node)
        
        # _for_loop -> affine.for + recursive visit
        if target is hl_tracing_ops._for_loop:
            return self.visit_for_loop(node)
        
        # _phi -> helion.phi
        if target is hl_tracing_ops._phi:
            return self.visit_phi(node)
        
        # _new_var -> pass through
        if target is hl_tracing_ops._new_var:
            return self.visit_new_var(node)
        
        # _host_tensor -> function argument mapping
        if target is hl_tracing_ops._host_tensor:
            return self.visit_host_tensor(node)
        
        # aten.sym_size.int -> inline concrete value
        if target is aten.sym_size.int:
            return self.visit_sym_size(node)
        
        # load -> tensor.extract_slice
        if target is hl_memory_ops.load:
            return self.visit_load(node)
        
        # store -> tensor.insert_slice
        if target is hl_memory_ops.store:
            return self.visit_store(node)
        
        # tile_id / tile_begin / tile_end -> IV-based SSA
        if target is hl_tile_ops.tile_id:
            return self.visit_tile_id(node)
        if target is hl_tile_ops.tile_begin:
            return self.visit_tile_begin(node)
        if target is hl_tile_ops.tile_end:
            return self.visit_tile_end(node)

        # _mask_to -> shortcircuit (pass through input, boundary check placeholder)
        if target is hl_tracing_ops._mask_to:
            return self.visit_mask_to(node)
        
        # subscript -> tensor.extract_slice / tensor.expand_shape
        if target is hl_view_ops.subscript:
            return self.visit_subscript(node)
        
        # getitem -> map to loop result
        if target is getattr(__builtins__, 'getitem', None) or \
           (hasattr(target, '__name__') and target.__name__ == 'getitem'):
            return self.visit_getitem(node)
        
        # Check for operator.getitem
        import operator
        if target is operator.getitem:
            return self.visit_getitem(node)
        
        # aten.full.default -> tensor.empty + linalg.fill
        if target is aten.full.default:
            return self.visit_aten_full(node)
        
        # aten.* compute operations - emit via torch-mlir
        if hasattr(target, '__module__') and 'aten' in str(target):
            return self.visit_aten_compute(node)
        
        # Unsupported target
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
        self.ctx.node_values[node.name] = ssa
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
        import re
        from helion._compiler.variable_origin import BlockSizeOrigin

        raw_id: int | None = None

        sym_val = node.meta.get("val")
        if sym_val is not None and hasattr(sym_val, '_sympy_'):
            sym = sym_val._sympy_()
            host_function = self.ctx.bound_kernel.host_function
            origin_info = host_function.expr_to_origin.get(sym)
            if origin_info and isinstance(origin_info.origin, BlockSizeOrigin):
                raw_id = origin_info.origin.block_id

        if raw_id is None:
            match = re.search(r'block_size_(\d+)', node.name)
            if match:
                raw_id = int(match.group(1))

        if raw_id is not None:
            return self.ctx.resolve_block_id(raw_id)
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
    
    def visit_for_loop(self, node: fx.Node) -> str:

        """Generate affine.for and visit the inner ForLoopGraphInfo."""
        graph_id = node.args[0]
        args = node.args[3]   # [acc]
        
        # Get the ForLoopGraphInfo
        for_graph = self.ctx.graphs.get(graph_id)
        if for_graph is None:
            raise ValueError(f"ForLoopGraphInfo with graph_id={graph_id} not registered")
        
        # Get block_ids from the graph (resolve through alias map)
        block_ids = getattr(for_graph, 'block_ids', [self.loop_depth])
        block_id = self.ctx.resolve_block_id(block_ids[0] if block_ids else 2)
        
        # Use pre-computed trip count (computed outside affine.parallel)
        trip_count_ssa = self.ctx.reduction_trip_counts.get(block_id)
        if trip_count_ssa is None:
            raise ValueError("No trip count found for loop")
        
        # Examine ForLoopGraphInfo output to determine actual loop-carried values
        # Find the output node to get the number of yielded values
        output_nodes = [n for n in for_graph.graph.nodes if n.op == "output"]
        num_loop_outputs = 0
        if output_nodes:
            output_args = output_nodes[0].args[0]
            if isinstance(output_args, (list, tuple)):
                num_loop_outputs = len(output_args)
            elif output_args is not None:
                num_loop_outputs = 1
        
        # ---------------------------------------------------------------------
        # Collect all arguments and classify them using phi-node analysis:
        # - Read-only args: passed to the loop but NOT yielded back
        # - Loop-carried args (iter_args): yielded and updated each iteration
        #
        # We determine which args are loop-carried by scanning the _for_loop
        # node's users for getitem -> _phi chains. An arg is loop-carried iff
        # it appears as the init value (lhs) of a _phi node whose rhs is a
        # getitem of this _for_loop's result.
        # ---------------------------------------------------------------------
        import operator as op_module
        iter_arg_names = set()
        for user in node.users:
            if user.target is op_module.getitem:
                for phi_user in user.users:
                    if hasattr(phi_user, 'target') and phi_user.target is hl_tracing_ops._phi:
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

        # Classify based on phi analysis
        readonly_args_info = []
        iter_args_info = []
        for info_tuple in all_args_info:
            name, ssa, fx_name = info_tuple
            if fx_name and fx_name in iter_arg_names:
                iter_args_info.append(info_tuple)
            else:
                readonly_args_info.append(info_tuple)
        
        # Determine tensor type for iter_args from their initial values
        iter_args_types = []
        for name, ssa, fx_name in iter_args_info:
            if fx_name and fx_name in self.ctx.node_types:
                iter_args_types.append(self.ctx.node_types[fx_name])
            else:
                raise RuntimeError(f"Cannot compute MLIR type for node {fx_name}")
        
        # Emit affine.for
        iv = f"%iv_block_{block_id}"
        result = self.mlir_output_helper.fresh(f"for_result_{graph_id}")
        
        # Build iter_args string
        iter_args_parts = []
        for (name, ssa, _), _ in zip(iter_args_info, iter_args_types):
            iter_args_parts.append(f"%{name} = {ssa}")
        iter_args_str = ", ".join(iter_args_parts)
        
        # Result types
        result_types = ", ".join(iter_args_types)
        
        # For multi-value results, use :N syntax (e.g., %result:3 = ...)
        num_results = len(iter_args_info)
        if num_results > 1:
            result_binding = f'{result}:{num_results}'
        else:
            result_binding = result
        
        self.mlir_output_helper.emit(
            f'{result_binding} = affine.for {iv} = 0 to {trip_count_ssa} '
            f'iter_args({iter_args_str}) -> ({result_types}) {{'
        )
        self.mlir_output_helper.push()
        
        # Set up args inside loop - map placeholder names to appropriate SSA values
        # Device IR uses naming patterns like arg0_1, arg1_1, etc. corresponding
        # positionally to the args list. We map each placeholder to either:
        # - Read-only: original SSA from outer scope
        # - iter_args: the loop block argument SSA (%acc_iterN)
        old_loop_iter_args = self.loop_iter_args.copy()

        iter_arg_idx = 0
        for i, a in enumerate(args):
            placeholder_name = f"arg{i}_1"
            if isinstance(a, fx.Node) and a.name in iter_arg_names:
                # Loop-carried iter_arg -> map to block argument
                iter_name = iter_args_info[iter_arg_idx][0]
                self.loop_iter_args[placeholder_name] = f"%{iter_name}"
                # Propagate type from init value to iter_arg placeholder
                if a.name in self.ctx.node_types:
                    self.ctx.node_types[placeholder_name] = self.ctx.node_types[a.name]
                iter_arg_idx += 1
            elif isinstance(a, fx.Node):
                # Read-only -> use original SSA from outer scope
                ssa = self.ctx.node_values.get(a.name, f"%{a.name}")
                self.loop_iter_args[placeholder_name] = ssa
                # Propagate type from outer scope to inner placeholder
                if a.name in self.ctx.node_types:
                    self.ctx.node_types[placeholder_name] = self.ctx.node_types[a.name]

        
        self.loop_depth += 1
        
        # Set current block_id for visit_load to reference the correct IV
        old_block_id = self.current_block_id
        self.current_block_id = block_id
        
        # Visit inner graph
        self.visit_graph(for_graph)
        
        # Restore old block_id
        self.current_block_id = old_block_id
        
        self.loop_depth -= 1
        
        # Restore loop_iter_args
        self.loop_iter_args = old_loop_iter_args
        
        # Emit yield with ALL results from inner graph
        if isinstance(self.current_loop_result, list) and len(self.current_loop_result) > 1:
            # Multiple yield values
            yield_values = ", ".join(self.current_loop_result)
            yield_types = ", ".join(iter_args_types)
            self.mlir_output_helper.emit(f'affine.yield {yield_values} : {yield_types}')
        else:
            # Single yield value (backward compatible)
            yield_value = self.current_loop_result[0] if isinstance(self.current_loop_result, list) else (self.current_loop_result or f"%{iter_args_info[0][0]}")
            yield_type = iter_args_types[0] if iter_args_types else f"tensor<?x?xf32>"
            self.mlir_output_helper.emit(f'affine.yield {yield_value} : {yield_type}')
        
        self.mlir_output_helper.pop()
        self.mlir_output_helper.emit("}")
        
        # Store the loop result - for multi-value loops, this SSA represents all results
        # Individual results are extracted via getitem
        self.ctx.node_values[node.name] = result
        
        # Also store the result SSAs for each output index
        # This allows visit_getitem to extract individual results
        if isinstance(self.current_loop_result, list) and len(self.current_loop_result) > 1:
            self.ctx.loop_result_values = {
                node.name: result,  # The tuple result SSA
                '_count': len(self.current_loop_result)  # Number of results
            }
        
        return result

    
    def visit_phi(self, node: fx.Node) -> str:
        """Handle phi nodes for loop-carried value merging.
        
        For Helion Device IR, _phi merges the initial value with the loop result.
        Since MLIR's affine.for already handles iter_args merge semantics,
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
            # No helion.phi needed - MLIR's affine.for handles the merge
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
            ssa = self.ctx.node_values.get(arg.name, f"%{arg.name}")
        else:
            ssa = str(arg)
        
        self.ctx.node_values[node.name] = ssa
        if isinstance(arg, fx.Node):
            self.ctx.node_types[node.name] = self._get_tensor_type(arg)
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
        
        def parse_memref_dimensions(type_str: str) -> list[str]:
            if 'memref<' in type_str and '>' in type_str:
                content = type_str[type_str.find('<')+1 : type_str.rfind('>')]
                if 'x' in content:
                    return content.split('x')[:-1]
            return []
        
        src_dims = parse_memref_dimensions(memref_type)
        
        for i, idx in enumerate(indices):
            if isinstance(idx, slice):
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
                
            elif isinstance(idx, fx.Node):
                # Get the SSA value for this index (tile size)
                idx_ssa = self.ctx.node_values.get(idx.name, f"%{idx.name}")
                
                # Check if this index has a BlockSizeOrigin - if so, it represents a tile size
                # and we need to compute offset as iv * block_size
                block_id = self._try_get_block_id_from_node(idx)
                
                if block_id is not None:
                    # This index represents a tile size (either block_size_X or sym_size_int with BlockSizeOrigin)
                    # The offset comes from the corresponding loop IV
                    iv_ssa = f"%iv_block_{block_id}"
                    
                    # Compute offset: iv * block_size (always dynamic since iv is runtime)
                    offset_ssa = self.mlir_output_helper.fresh("offset")
                    self.mlir_output_helper.emit(
                        f'{offset_ssa} = arith.muli {iv_ssa}, {idx_ssa} : index'
                    )
                    offsets.append((offset_ssa, False))
                    sizes.append((idx_ssa, False))
                    output_dim_sizes.append(None)  # Dynamic (block sizes are runtime)
                else:
                    # Generic index - treat as offset with size 1
                    offsets.append((idx_ssa, False))
                    sizes.append(("1", True))
                    output_dim_sizes.append(1)
                    
            elif isinstance(idx, int):
                # Integer literal offset with size 1
                offsets.append((str(idx), True))
                sizes.append(("1", True))
                output_dim_sizes.append(1)
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
        def compute_strides_from_dims(dims: list[str]) -> list[int] | None:
            """Compute row-major strides from dimension strings. Returns None if any dim is dynamic."""
            try:
                int_dims = [int(d) for d in dims]
            except ValueError:
                return None  # Has dynamic dimensions
            
            strides = []
            product = 1
            for d in reversed(int_dims):
                strides.append(product)
                product *= d
            return list(reversed(strides))
        
        src_strides = compute_strides_from_dims(src_dims)
        has_dynamic_offset = any(not is_static for _, is_static in offsets)
        
        if output_dims:
            if src_strides is not None and has_dynamic_offset:
                # Need strided layout with offset: ? for dynamic offsets from static source
                strides_str = ", ".join(str(s) for s in src_strides)
                subview_type = f"memref<{'x'.join(output_dims)}x{dtype_str}, strided<[{strides_str}], offset: ?>>"
            else:
                subview_type = f"memref<{'x'.join(output_dims)}x{dtype_str}>"
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
        
        def parse_type_dimensions(type_str: str) -> list[str]:
            if ('<' in type_str and '>' in type_str):
                content = type_str[type_str.find('<')+1 : type_str.rfind('>')]
                if 'x' in content:
                    return content.split('x')[:-1]
            return []
        
        src_dims = parse_type_dimensions(value_type)
        
        for i, idx in enumerate(indices):
            if isinstance(idx, slice):
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
                
            elif isinstance(idx, fx.Node):
                idx_ssa = self.ctx.node_values.get(idx.name, f"%{idx.name}")
                
                # Check if this index has a BlockSizeOrigin - if so, it represents a tile size
                # and we need to compute offset as iv * block_size
                block_id = self._try_get_block_id_from_node(idx)
                
                if block_id is not None:
                    # This index represents a tile size (either block_size_X or sym_size_int with BlockSizeOrigin)
                    # The offset comes from the corresponding loop IV
                    iv_ssa = f"%iv_block_{block_id}"
                    
                    offset_ssa = self.mlir_output_helper.fresh("offset")
                    self.mlir_output_helper.emit(f'{offset_ssa} = arith.muli {iv_ssa}, {idx_ssa} : index')
                    offsets.append((offset_ssa, False))
                    sizes.append((idx_ssa, False))
                    output_dim_sizes.append(None)  # Dynamic (block sizes are runtime)
                else:
                    # Generic index
                    offsets.append((idx_ssa, False))
                    sizes.append(("1", True))
                    output_dim_sizes.append(1)
                    
            elif isinstance(idx, int):
                # Integer literal offset with size 1
                offsets.append((str(idx), True))
                sizes.append(("1", True))
                output_dim_sizes.append(1)
            else:
                # Unknown type - default to 0 offset, size 1
                offsets.append(("0", True))
                sizes.append(("1", True))
                output_dim_sizes.append(1)
        
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
        
        # Helper to extract dimensions from memref type string
        def parse_memref_dimensions(type_str: str) -> list[str]:
            """Extract dimension strings from a memref type like 'memref<256x128xf32>'."""
            if 'memref<' in type_str and '>' in type_str:
                content = type_str[type_str.find('<')+1 : type_str.rfind('>')]
                # Handle optional layout info by splitting on comma first
                if ',' in content:
                    content = content.split(',')[0]
                if 'x' in content:
                    return content.split('x')[:-1]  # Exclude dtype
            return []
        
        memref_dims = parse_memref_dimensions(memref_type)
        
        # Compute strides from source memref dimensions (row-major)
        # For source memref<D0xD1xD2xf32>, strides are [D1*D2, D2, 1]
        def compute_strides_from_dims(dims: list[str]) -> list[int] | None:
            """Compute row-major strides from dimension strings. Returns None if any dim is dynamic."""
            try:
                int_dims = [int(d) for d in dims]
            except ValueError:
                return None  # Has dynamic dimensions
            
            strides = []
            product = 1
            for d in reversed(int_dims):
                strides.append(product)
                product *= d
            return list(reversed(strides))
        
        src_strides = compute_strides_from_dims(memref_dims)
        has_dynamic_offset = any(not is_static for _, is_static in offsets)

        # Rank-reducing subview: drop size-1 dims that come from scalar indices
        # so the subview rank matches the value tensor rank.
        reduced_dims = []
        reduced_strides = []
        for i, dim in enumerate(output_dims):
            if dim == "1" and output_dim_sizes[i] == 1:
                continue  # scalar index → drop this dimension
            reduced_dims.append(dim)
            if src_strides is not None:
                reduced_strides.append(src_strides[i])

        if reduced_dims:
            if src_strides is not None and has_dynamic_offset:
                strides_layout_str = ", ".join(str(s) for s in reduced_strides)
                subview_type = f"memref<{'x'.join(reduced_dims)}x{dtype_str}, strided<[{strides_layout_str}], offset: ?>>"
            else:
                subview_type = f"memref<{'x'.join(reduced_dims)}x{dtype_str}>"
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
    
    def visit_getitem(self, node: fx.Node) -> str:
        """Map getitem to the corresponding loop result.
        
        For multi-value affine.for loops, getitem(_for_loop, i) extracts
        the i-th return value as %result#i syntax.
        """
        source = node.args[0]
        index = node.args[1]
        
        # For _for_loop results, getitem extracts the i-th return value
        source_ssa = self.ctx.node_values.get(source.name, f"%{source.name}") if isinstance(source, fx.Node) else str(source)
        
        # Check if this is extracting from a multi-value loop result
        if isinstance(source, fx.Node) and source.name in self.ctx.loop_result_values:
            # Multi-value loop result - use #index syntax
            result_ssa = f"{source_ssa}#{index}"
            self.ctx.node_values[node.name] = result_ssa
            
            # Also register the tensor type from FakeTensor metadata
            if 'val' in node.meta:
                tensor_type = self.ctx.compute_mlir_type_from_fake_tensor(node.meta['val'])
                self.ctx.node_types[node.name] = tensor_type
            
            return result_ssa
        
        # For single-return loops, just use the result directly
        self.ctx.node_values[node.name] = source_ssa
        return source_ssa

    
    def visit_tile_id(self, node: fx.Node) -> str:
        """Map tile_id(block_size_X) to the corresponding parallel loop IV."""
        block_size_node = node.args[0]
        block_id = self._try_get_block_id_from_node(block_size_node)
        if block_id is None:
            raise ValueError(f"Cannot determine block_id for tile_id node {node.name}")
        iv_ssa = f"%iv_block_{block_id}"
        self.ctx.node_values[node.name] = iv_ssa
        return iv_ssa

    def visit_tile_begin(self, node: fx.Node) -> str:
        """Map tile_begin(block_size_X) to IV * block_size."""
        block_size_node = node.args[0]
        block_id = self._try_get_block_id_from_node(block_size_node)
        if block_id is None:
            raise ValueError(f"Cannot determine block_id for tile_begin node {node.name}")
        canonical_id = self.ctx.resolve_block_id(block_id)
        iv_ssa = f"%iv_block_{canonical_id}"
        bs_ssa = self.ctx.block_size_ssa[canonical_id]
        result = self.mlir_output_helper.fresh("tile_begin")
        self.mlir_output_helper.emit(f'{result} = arith.muli {iv_ssa}, {bs_ssa} : index')
        self.ctx.node_values[node.name] = result
        return result

    def visit_tile_end(self, node: fx.Node) -> str:
        """Map tile_end(block_size_X) to (IV + 1) * block_size."""
        block_size_node = node.args[0]
        block_id = self._try_get_block_id_from_node(block_size_node)
        if block_id is None:
            raise ValueError(f"Cannot determine block_id for tile_end node {node.name}")
        canonical_id = self.ctx.resolve_block_id(block_id)
        iv_ssa = f"%iv_block_{canonical_id}"
        bs_ssa = self.ctx.block_size_ssa[canonical_id]
        one = self.mlir_output_helper.fresh("one")
        self.mlir_output_helper.emit(f'{one} = arith.constant 1 : index')
        iv_plus_one = self.mlir_output_helper.fresh("iv_plus_one")
        self.mlir_output_helper.emit(f'{iv_plus_one} = arith.addi {iv_ssa}, {one} : index')
        result = self.mlir_output_helper.fresh("tile_end")
        self.mlir_output_helper.emit(f'{result} = arith.muli {iv_plus_one}, {bs_ssa} : index')
        self.ctx.node_values[node.name] = result
        return result

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
            ssa = self.ctx.node_values.get(tensor_node.name, f"%{tensor_node.name}")
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
        
        needs_slicing = False
        # Always slice to resolve dynamic dimensions or if explicit indices provided
        if slice_indices:
             needs_slicing = True
        
        extracted_ssa = current_ssa
        # Default intermediate type is source type
        extracted_type = source_type
        
        if needs_slicing:
            offsets_ssa = []
            sizes_ssa = []
            strides_ssa = []
            
            input_dim_idx = 0
            
            # Get FakeTensor from source node for dimension resolution
            source_fake_tensor = tensor_node.meta.get("val") if isinstance(tensor_node, fx.Node) else None
            
            for idx in slice_indices:
                if isinstance(idx, slice):
                    # Handle slice(None) aka [:]
                    # Offset 0 - use inline literal
                    offsets_ssa.append("0")
                    
                    # Size: try to resolve from FakeTensor
                    dim_resolved = None
                    if source_fake_tensor is not None and input_dim_idx < source_fake_tensor.ndim:
                        dim_size = source_fake_tensor.size(input_dim_idx)
                        value_str, is_static = self.resolve_dimension(dim_size, input_dim_idx)
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
            
            res_rank = 0
            for idx in slice_indices:
                if isinstance(idx, slice):
                    res_rank += 1
            
            # Derive dtype from source FakeTensor
            source_dtype = self._get_element_type_from_node(tensor_node)
            dims_str = "x".join(["?"] * res_rank)
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
        
        # -----------------------------------------------------------
        # Step 2: Handle New Axis (tensor.expand_shape)
        # -----------------------------------------------------------
        
        has_newaxis = any(idx is None for idx in indices)
        
        if has_newaxis:
            # -----------------------------------------------------------------
            # Build reassociation map for tensor.expand_shape
            # -----------------------------------------------------------------
            # tensor.expand_shape requires a reassociation map that groups
            # output dimensions back to input dimensions. For newaxis (None):
            # - Each None adds a size-1 dimension that didn't exist in input
            # - These new dims must be grouped with real dims from input
            # 
            # Strategy: attach each 'new' dimension to the NEXT real dimension
            # in the output. If no next real dim, attach to the last one.
            #
            # Example: indices = [slice, None, slice, None]
            #   output_dim_types = ['real', 'new', 'real', 'new']
            #   input_dim_assignments: {0: [0], 1: [1, 2, 3]} (expanded: [[0], [1,2,3]])
            # -----------------------------------------------------------------
            output_dim_types = [] 
            for idx in indices:
                if isinstance(idx, slice):
                    output_dim_types.append('real')
                elif idx is None:
                    output_dim_types.append('new')
                # ints are ignored/dropped
            
            input_dim_assignments = {} # int -> list[int]
            
            # Pass 1: map 'real' output dims to input dims
            real_dim_counter = 0
            real_output_indices = []
            for i, dtype in enumerate(output_dim_types):
                if dtype == 'real':
                    input_dim_assignments[real_dim_counter] = [i]
                    real_output_indices.append(i)
                    real_dim_counter += 1
            
            # Pass 2: distribute 'new' dims
            for i, dtype in enumerate(output_dim_types):
                if dtype == 'new':
                    # Attach to NEXT real dim group
                    found_next = False
                    target_input_dim = -1
                    
                    for k in range(real_dim_counter):
                        # If the real dim associated with input K is AFTER current 'new' dim i
                        if input_dim_assignments[k][0] > i:
                            target_input_dim = k
                            break
                    
                    if target_input_dim != -1:
                        input_dim_assignments[target_input_dim].insert(0, i) # Prepend to that group
                        found_next = True
                    
                    if not found_next:
                        # Attach to last input dim
                        last_input = real_dim_counter - 1
                        if last_input >= 0:
                            input_dim_assignments[last_input].append(i)
            
            # Flatten to list
            reassoc_list = []
            for k in range(real_dim_counter):
                reassoc_list.append(input_dim_assignments[k])
                
            reassoc_str = "[" + ", ".join(["[" + ", ".join(map(str, grp)) + "]" for grp in reassoc_list]) + "]"
            
            # Result type - use '1' for new dimensions (from None), '?' for real dimensions
            result_dims = []
            for dtype in output_dim_types:
                if dtype == 'new':
                    result_dims.append("1")  # New dimensions from None are always size 1
                else:
                    result_dims.append("?")  # Real dimensions are dynamic
            
            source_dtype = self._get_element_type_from_node(tensor_node)
            dims_str = "x".join(result_dims)
            result_type = f"tensor<{dims_str}x{source_dtype}>"
            
            result_expand = self.mlir_output_helper.fresh("expand")
            
            # Compute output_shape for dynamic dimensions
            # For each output dimension, we need its size as an SSA value or literal
            # We use the source tensor's FakeTensor to resolve dimensions
            output_shape_ssas = []
            extracted_dim_idx = 0  # Track which dimension of extracted_ssa we're on
            
            # Get source FakeTensor for dimension resolution
            source_fake_tensor = tensor_node.meta.get("val") if isinstance(tensor_node, fx.Node) else None
            
            # Build mapping from extracted dims to source dims (accounting for integer indexing)
            # slice_indices already excludes None, so:
            # - Each slice contributes one dim (maps to source dim)
            # - Each int contributes one dim that is size 1 (rank-reducing)
            source_dim_for_extracted = []
            src_dim = 0
            for idx in slice_indices:
                if isinstance(idx, slice):
                    source_dim_for_extracted.append(src_dim)
                    src_dim += 1
                elif isinstance(idx, int) or isinstance(idx, fx.Node):
                    # Integer indexing - size is 1, but still consumes a source dim
                    source_dim_for_extracted.append(src_dim)
                    src_dim += 1
            
            for i, dtype in enumerate(output_dim_types):
                if dtype == 'new':
                    # New dimension from None - always size 1 (inline literal)
                    output_shape_ssas.append("1")
                else:
                    # Real dimension - try to resolve from source FakeTensor
                    dim_resolved = None
                    if source_fake_tensor is not None and extracted_dim_idx < len(source_dim_for_extracted):
                        source_dim = source_dim_for_extracted[extracted_dim_idx]
                        if source_dim < source_fake_tensor.ndim:
                            dim_size = source_fake_tensor.size(source_dim)
                            value_str, is_static = self.resolve_dimension(dim_size, source_dim)
                            if value_str is not None:
                                dim_resolved = value_str
                    
                    if dim_resolved is None:
                        # Fallback to tensor.dim
                        dim_idx_ssa = self.mlir_output_helper.fresh("dim_idx")
                        dim_ssa = self.mlir_output_helper.fresh("dim")
                        self.mlir_output_helper.emit(f'{dim_idx_ssa} = arith.constant {extracted_dim_idx} : index')
                        self.mlir_output_helper.emit(f'{dim_ssa} = tensor.dim {extracted_ssa}, {dim_idx_ssa} : {extracted_type}')
                        dim_resolved = dim_ssa
                    
                    output_shape_ssas.append(dim_resolved)
                    extracted_dim_idx += 1
            
            output_shape_str = "[" + ", ".join(output_shape_ssas) + "]"
            
            self.mlir_output_helper.emit(
                f'{result_expand} = tensor.expand_shape {extracted_ssa} {reassoc_str} '
                f'output_shape {output_shape_str} : '
                f'{extracted_type} into {result_type}'
            )
            
            self.ctx.node_values[node.name] = result_expand
            return result_expand
            
        else:
            self.ctx.node_values[node.name] = extracted_ssa
            return extracted_ssa
    
    def visit_aten_compute(self, node: fx.Node) -> str:
        """Generate MLIR for ATen compute ops using torch-mlir.
        
        Uses torch-mlir's FxImporter to generate MLIR for ATen operations.
        The output is always lowered to linalg-on-tensors.
        
        Optimizes tensor.dim operations by passing pre-existing dimension SSAs.
        """
        from helion._compiler.variable_origin import BlockSizeOrigin
        target = node.target
        
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
        
        # Use torch-mlir to generate MLIR for this operation
        mlir_text = import_aten_node_to_mlir(node)
        if mlir_text is None:
             raise RuntimeError(f"Failed to lower ATen op: {node.name} ({target})")

        
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

        # Build operand_types map for type consistency
        op_types: dict[str, str] = {}
        for op_ssa, op_node in zip(tensor_operands, tensor_operand_nodes):
            if isinstance(op_node, fx.Node):
                try:
                    op_types[op_ssa] = self._get_tensor_type(op_node)
                except RuntimeError:
                    pass

        from .torch_mlir_helper import inline_torch_mlir_output
        result = inline_torch_mlir_output(
            mlir_text,
            tensor_operands,
            self.mlir_output_helper,
            dimension_ssa_map=dimension_ssa_map if dimension_ssa_map else None,
            scalar_operand_map=scalar_operand_map if scalar_operand_map else None,
            operand_types=op_types if op_types else None,
        )
        
        self.ctx.node_values[node.name] = result
        return result
    
    def _get_element_type_from_node(self, node: fx.Node) -> str:
        """Extract MLIR element type string from a node's FakeTensor dtype."""
        from .mlir_utils import torch_dtype_to_mlir_element_type
        if isinstance(node, fx.Node):
            fake = node.meta.get("val")
            if fake is not None and hasattr(fake, "dtype"):
                return torch_dtype_to_mlir_element_type(fake.dtype)
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
            return self.ctx.compute_mlir_type_from_fake_tensor(fake_tensor)
        
        raise RuntimeError(f"Cannot compute MLIR type for node {name}")
    
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
