"""IR Visitor for walking Device IR graphs and generating MLIR.

This module implements a visitor pattern for converting Helion Device IR
to MLIR by walking FX graph nodes instruction-by-instruction.

The visitor dispatches to specific handlers based on the node's target:
- _get_symnode -> loom.get_symbol
- full -> tensor.empty + linalg.fill
- _for_loop -> affine.for + recursive visit
- _phi -> helion.phi
- _host_tensor -> function argument mapping
- aten.sym_size.int -> inline concrete value
- load -> tensor.extract_slice
- store -> tensor.insert_slice
- aten.* compute -> torch.aten.* (via torch-mlir dialect)
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

from .mlir_builder import (
    format_attr_dict,
    format_string_attr,
    torch_dtype_to_mlir_element_type,
    format_tensor_type,
)
from .torch_mlir_helper import (
    TorchMLIRNodeImporter,
    get_aten_op_info,
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
        self.builder = ctx.builder
        
        # Loop-local state (managed per loop, not persisted globally)
        # These are reset/managed during visit_for_loop calls
        self.loop_iter_args: dict[str, str] = {}  # placeholder name → SSA (per loop context)
        self.current_loop_result: str | list[str] | None = None  # Set by inner graph output
        self.loop_depth: int = 0  # Depth tracking for nested loops
        self.current_block_id: int | None = None  # Current loop's block_id for IV reference
        
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
        
        # Fallback for unknown targets
        return self.visit_unknown(node)

    
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
        ssa = self.builder.fresh(node.name)
        self.ctx.node_values[node.name] = ssa
        return ssa
    
    # -------------------------------------------------------------------------
    # Specific Node Handlers
    # -------------------------------------------------------------------------
    
    def visit_get_symnode(self, node: fx.Node) -> str:
        """Look up pre-emitted SSA for symnode references.
        
        Resolution strategy based on Origin type:
        1. BlockSizeOrigin -> Use pre-emitted ctx.block_size_ssa[block_id]
        2. Other origins -> Use shape_env.var_to_val for concrete value -> arith.constant
        """
        from helion._compiler.variable_origin import BlockSizeOrigin
        
        # Get sympy expression from node metadata
        sym_val = node.meta.get("val")
        if sym_val is None:
            raise ValueError(f"No meta['val'] for symnode {node.name}")
        
        sym = sym_val._sympy_()
        
        # Look up origin info from the host_function
        host_function = self.ctx.bound_kernel.host_function
        origin_info = host_function.expr_to_origin.get(sym)
        origin = origin_info.origin if origin_info else None
        
        # 1. BlockSizeOrigin -> use pre-emitted block_size_ssa
        if isinstance(origin, BlockSizeOrigin):
            block_id = origin.block_id
            ssa = self.ctx.block_size_ssa.get(block_id)
            if ssa is None:
                raise ValueError(f"No block_size_ssa for block_id={block_id}")
            self.ctx.node_values[node.name] = ssa
            return ssa
        
        # 2. Other origins -> use shape_env for concrete value
        shape_env = self.ctx.bound_kernel.env.shape_env
        if sym in shape_env.var_to_val:
            concrete_val = int(shape_env.var_to_val[sym])
            ssa = self.builder.fresh(node.name.replace(".", "_"))
            self.builder.emit(f'{ssa} = arith.constant {concrete_val} : index')
            self.ctx.node_values[node.name] = ssa
            return ssa
        
        # Fallback: no resolution found
        raise ValueError(
            f"Cannot resolve symbol {sym} for {node.name}. "
            f"Origin: {origin}, not in shape_env.var_to_val"
        )
    
    def visit_full(self, node: fx.Node) -> str:
        """Generate tensor.empty + linalg.fill for tensor initialization.
        
        Replaces the custom helion.full with standard MLIR dialects:
        - tensor.empty to create an uninitialized tensor
        - linalg.fill to fill it with the specified value
        
        For special values like -inf, we emit arith.constant with IEEE 754 hex 
        representation.
        """
        import math
        
        shape_nodes = node.args[0]  # List of FX nodes or values
        fill_value = node.args[1] if len(node.args) > 1 else 0.0
        dtype = node.args[2] if len(node.args) > 2 else torch.float32
        
        # Resolve shape to SSA values - integer literals need to become arith.constant
        shape_ssa = []
        for s in shape_nodes:
            if isinstance(s, fx.Node):
                shape_ssa.append(self.ctx.node_values.get(s.name, f"%{s.name}"))
            elif isinstance(s, int):
                # Integer literals need to be emitted as arith.constant
                const_ssa = self.builder.fresh("dim")
                self.builder.emit(f'{const_ssa} = arith.constant {s} : index')
                shape_ssa.append(const_ssa)
            else:
                # Fallback for other values
                shape_ssa.append(str(s))
        
        dtype_str = torch_dtype_to_mlir_element_type(dtype) if dtype else "f32"
        
        # Use FakeTensor metadata to determine tensor type with correct dimensions
        # This preserves concrete dimensions (e.g., 128 for head_dim) instead of using '?' for all
        fake_tensor = node.meta.get("val")
        if fake_tensor is not None and hasattr(fake_tensor, "shape"):
            tensor_type = self.ctx.compute_mlir_type_from_fake_tensor(fake_tensor, dtype_str)
        else:
            # Fallback: all dynamic dimensions
            dim_wildcards = "x".join(["?"] * len(shape_ssa))
            tensor_type = f"tensor<{dim_wildcards}x{dtype_str}>"
        
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
        
        empty_ssa = self.builder.fresh("empty")
        shape_str = ", ".join(dynamic_shape_ssa)
        self.builder.emit(f'{empty_ssa} = tensor.empty({shape_str}) : {tensor_type}')
        
        # Step 2: Emit fill value constant
        fill_val_ssa = self.builder.fresh("fill_val")
        if isinstance(fill_value, float) and (math.isinf(fill_value) or math.isnan(fill_value)):
            # Handle special float values with IEEE 754 hex representation
            if math.isinf(fill_value):
                if fill_value < 0:
                    hex_val = "0xFF800000"  # -inf for f32
                else:
                    hex_val = "0x7F800000"  # +inf for f32
            else:
                hex_val = "0x7FC00000"  # nan for f32
            self.builder.emit(f'{fill_val_ssa} = arith.constant {hex_val} : {dtype_str}')
        else:
            # Regular float value
            self.builder.emit(f'{fill_val_ssa} = arith.constant {fill_value} : {dtype_str}')
        
        # Step 3: Emit linalg.fill
        filled_ssa = self.builder.fresh("filled")
        self.builder.emit(
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
                const_ssa = self.builder.fresh("dim")
                self.builder.emit(f'{const_ssa} = arith.constant {s} : index')
                shape_ssa.append(const_ssa)
            else:
                # Fallback for other values
                shape_ssa.append(str(s))
        
        dtype_str = torch_dtype_to_mlir_element_type(dtype) if dtype else "f32"
        
        # Use FakeTensor metadata to determine tensor type with correct dimensions
        # This preserves concrete dimensions (e.g., 128 for head_dim) instead of using '?' for all
        fake_tensor = node.meta.get("val")
        if fake_tensor is not None and hasattr(fake_tensor, "shape"):
            tensor_type = self.ctx.compute_mlir_type_from_fake_tensor(fake_tensor, dtype_str)
        else:
            # Fallback: all dynamic dimensions
            dim_wildcards = "x".join(["?"] * len(shape_ssa))
            tensor_type = f"tensor<{dim_wildcards}x{dtype_str}>"
        
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
        
        empty_ssa = self.builder.fresh("empty")
        shape_str = ", ".join(dynamic_shape_ssa)
        self.builder.emit(f'{empty_ssa} = tensor.empty({shape_str}) : {tensor_type}')
        
        # Step 2: Emit fill value constant
        fill_val_ssa = self.builder.fresh("fill_val")
        if isinstance(fill_value, float) and (math.isinf(fill_value) or math.isnan(fill_value)):
            # Handle special float values with IEEE 754 hex representation
            if math.isinf(fill_value):
                if fill_value < 0:
                    hex_val = "0xFF800000"  # -inf for f32
                else:
                    hex_val = "0x7F800000"  # +inf for f32
            else:
                hex_val = "0x7FC00000"  # nan for f32
            self.builder.emit(f'{fill_val_ssa} = arith.constant {hex_val} : {dtype_str}')
        else:
            # Regular float value
            self.builder.emit(f'{fill_val_ssa} = arith.constant {fill_value} : {dtype_str}')
        
        # Step 3: Emit linalg.fill
        filled_ssa = self.builder.fresh("filled")
        self.builder.emit(
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
        
        # Get block_ids from the graph
        block_ids = getattr(for_graph, 'block_ids', [self.loop_depth])
        block_id = block_ids[0] if block_ids else 2
        
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
        
        # Resolve iter_args - only include loop-carried values (those that are yielded)
        # Read-only values (passed but not yielded) should be handled separately
        all_args_info = []
        for i, a in enumerate(args):
            if isinstance(a, fx.Node):
                ssa = self.ctx.node_values.get(a.name, f"%{a.name}")
                all_args_info.append((f"acc_iter{i}", ssa, a.name))
            else:
                all_args_info.append((f"acc_iter{i}", str(a), None))
        
        # Split into loop-carried (last N that match output count) and read-only (first few)
        # The convention is: first args are read-only, last args are loop-carried
        if num_loop_outputs > 0 and num_loop_outputs < len(all_args_info):
            # First (len - num_outputs) are read-only, rest are loop-carried
            num_readonly = len(all_args_info) - num_loop_outputs
            readonly_args_info = all_args_info[:num_readonly]
            iter_args_info = all_args_info[num_readonly:]
        else:
            readonly_args_info = []
            iter_args_info = all_args_info
        
        # Determine tensor type for iter_args from their initial values
        iter_args_types = []
        for name, ssa, fx_name in iter_args_info:
            if fx_name and fx_name in self.ctx.node_types:
                iter_args_types.append(self.ctx.node_types[fx_name])
            else:
                # Fallback to f32 dynamic tensor if type is unknown
                # Use ctx.element_type for element type guess if possible
                iter_args_types.append(f"tensor<?x?xf32>")
        
        # Emit affine.for
        iv = f"%iv_block{block_id}"
        result = self.builder.fresh(f"for_result_{graph_id}")
        
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
        
        self.builder.emit(
            f'{result_binding} = affine.for {iv} = 0 to {trip_count_ssa} '
            f'iter_args({iter_args_str}) -> ({result_types}) {{'
        )
        self.builder.push()
        
        # Set up args inside loop - map placeholder names to appropriate SSA values
        old_loop_iter_args = self.loop_iter_args.copy()
        
        # Get node_args from ForLoopGraphInfo if available
        node_args = getattr(for_graph, 'node_args', [])
        
        # Map placeholders for read-only args (use original SSA value directly)
        for i, (_, ssa, _) in enumerate(readonly_args_info):
            placeholder_name = f"arg{i}_1"
            self.loop_iter_args[placeholder_name] = ssa  # Use original SSA
        
        # Map placeholders for loop-carried iter_args
        num_readonly = len(readonly_args_info)
        for i, (iter_name, _, _) in enumerate(iter_args_info):
            # Offset by number of read-only args
            placeholder_name = f"arg{num_readonly + i}_1"
            self.loop_iter_args[placeholder_name] = f"%{iter_name}"
            
            # Also map with different naming patterns used in Device IR
            self.loop_iter_args[f"arg0_{num_readonly + i + 1}"] = f"%{iter_name}"
        
        # For the first placeholder in loops - could be read-only or iter_arg
        if readonly_args_info:
            self.loop_iter_args["arg0_1"] = readonly_args_info[0][1]  # First read-only
        elif iter_args_info:
            self.loop_iter_args["arg0_1"] = f"%{iter_args_info[0][0]}"  # First iter_arg

        
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
            self.builder.emit(f'affine.yield {yield_values} : {yield_types}')
        else:
            # Single yield value (backward compatible)
            yield_value = self.current_loop_result[0] if isinstance(self.current_loop_result, list) else (self.current_loop_result or f"%{iter_args_info[0][0]}")
            yield_type = iter_args_types[0] if iter_args_types else f"tensor<?x?xf32>"
            self.builder.emit(f'affine.yield {yield_value} : {yield_type}')
        
        self.builder.pop()
        self.builder.emit("}")
        
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
        of a tensor argument. For example:
        - sym_size_int(arg0_1, 0) -> tensor.dim %tensor, 0 (dimension 0 of the tensor)
        - sym_size_int(arg0_1, 1) -> tensor.dim %tensor, 1 (dimension 1 of the tensor)
        
        The tensor_node (e.g., arg0_1) refers to a placeholder that corresponds to
        an argument passed to the ForLoopGraphInfo from the RootGraphInfo's _for_loop call.
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
        
        # Get the tensor type
        tensor_type = self._get_tensor_type(tensor_node) if isinstance(tensor_node, fx.Node) else f"tensor<?x?xf32>"
        
        # Emit tensor.dim to get the dimension size
        dim_idx_ssa = self.builder.fresh("dim_idx")
        self.builder.emit(f'{dim_idx_ssa} = arith.constant {dim} : index')
        
        result_ssa = self.builder.fresh("dim_size")
        self.builder.emit(f'{result_ssa} = tensor.dim {tensor_ssa}, {dim_idx_ssa} : {tensor_type}')
        
        self.ctx.node_values[node.name] = result_ssa
        return result_ssa
    
    def visit_load(self, node: fx.Node) -> str:
        """Generate tensor.extract_slice for tile loading.
        
        Replaces the custom helion.load with standard MLIR tensor.extract_slice.
        The tile sizes come from the load node's FakeTensor metadata (node.meta['val'].size()),
        and each dimension's SymInt origin is used to look up the pre-emitted SSA.
        """
        from helion._compiler.variable_origin import BlockSizeOrigin
        
        tensor_node = node.args[0]
        indices = node.args[1]  # [sym_size_int, block_size_2] or [block_size_0, block_size_1, slice(...)]
        
        tensor_ssa = self.ctx.node_values.get(tensor_node.name, f"%{tensor_node.name}")
        tensor_type = self._get_tensor_type(tensor_node)
        
        # Get output FakeTensor shape from node metadata to determine tile sizes
        output_fake_tensor = node.meta.get("val")
        host_function = self.ctx.bound_kernel.host_function
        shape_env = self.ctx.bound_kernel.env.shape_env
        
        # Helper: resolve a SymInt to SSA using Origin lookup
        def resolve_symint_to_ssa(sym_int, dim_hint: int) -> str:
            """Resolve a SymInt to its SSA value using Origin-based lookup."""
            sym = sym_int._sympy_()
            origin_info = host_function.expr_to_origin.get(sym)
            origin = origin_info.origin if origin_info else None
            
            # BlockSizeOrigin -> use pre-emitted block_size_ssa
            if isinstance(origin, BlockSizeOrigin):
                block_id = origin.block_id
                ssa = self.ctx.block_size_ssa.get(block_id)
                if ssa is not None:
                    return ssa
            
            # Other origins -> use shape_env for concrete value
            if sym in shape_env.var_to_val:
                concrete_val = int(shape_env.var_to_val[sym])
                ssa = self.builder.fresh(f"size_dim{dim_hint}")
                self.builder.emit(f'{ssa} = arith.constant {concrete_val} : index')
                return ssa
            
            raise ValueError(f"Cannot resolve SymInt {sym} for dimension {dim_hint}")
        
        # Collect offset and size SSA values for each dimension
        # Track which dimensions have static sizes (for result type construction)
        offsets_ssa = []
        sizes_ssa = []
        output_dim_sizes = []  # Store (static_size | None) for each dim
        
        for i, idx in enumerate(indices):
            if isinstance(idx, slice):
                # Full dimension slice - offset=0, size comes from tensor dim
                zero_ssa = self.builder.fresh("zero")
                self.builder.emit(f'{zero_ssa} = arith.constant 0 : index')
                offsets_ssa.append(zero_ssa)
                
                # Get dimension size from tensor
                # Check if source tensor's dimension is static
                src_dim_static = None
                if 'tensor<' in tensor_type:
                    type_content = tensor_type[tensor_type.find('<')+1 : tensor_type.rfind('>')]
                    if 'x' in type_content:
                        src_dims = type_content.split('x')[:-1]
                        if i < len(src_dims):
                            if src_dims[i] != '?':
                                try:
                                    src_dim_static = int(src_dims[i])
                                except ValueError:
                                    pass
                
                dim_ssa = self.builder.fresh("dim")
                dim_idx_ssa = self.builder.fresh("dim_idx")
                self.builder.emit(f'{dim_idx_ssa} = arith.constant {i} : index')
                self.builder.emit(f'{dim_ssa} = tensor.dim {tensor_ssa}, {dim_idx_ssa} : {tensor_type}')
                sizes_ssa.append(dim_ssa)
                output_dim_sizes.append(src_dim_static)  # Preserve static size
            elif isinstance(idx, fx.Node):
                # Get the SSA value for this index (offset)
                idx_ssa = self.ctx.node_values.get(idx.name, f"%{idx.name}")
                
                # Determine if this is a loop IV (sym_size_int) or a block size
                if 'sym_size' in idx.name:
                    # sym_size_int represents an offset derived from tensor.dim
                    # Use it directly as the offset
                    offsets_ssa.append(idx_ssa)
                    
                    # Size comes from the output FakeTensor's shape for this dimension
                    if output_fake_tensor is not None and i < output_fake_tensor.ndim:
                        dim_size = output_fake_tensor.size(i)
                        if hasattr(dim_size, '_sympy_'):
                            # It's a SymInt, resolve via Origin
                            size_ssa = resolve_symint_to_ssa(dim_size, i)
                            output_dim_sizes.append(None)  # Dynamic
                        else:
                            # Concrete int
                            size_ssa = self.builder.fresh(f"size_dim{i}")
                            self.builder.emit(f'{size_ssa} = arith.constant {int(dim_size)} : index')
                            output_dim_sizes.append(int(dim_size))  # Static
                        sizes_ssa.append(size_ssa)
                    else:
                        # Fallback: use idx_ssa as size (shouldn't happen with proper metadata)
                        sizes_ssa.append(idx_ssa)
                        output_dim_sizes.append(None)
                        
                elif 'block_size' in idx.name:
                    # block_size node indicates the tile size for this dimension
                    # The offset comes from the corresponding loop IV
                    
                    # Determine which block_id this index corresponds to
                    # For reduction loops, use current_block_id; for parallel, use position
                    if self.current_block_id is not None:
                        # Inside a reduction loop - use the current block's IV
                        iv_ssa = f"%iv_block{self.current_block_id}"
                    elif hasattr(self, 'current_loop_ivs') and i < len(self.current_loop_ivs):
                        iv_ssa = self.current_loop_ivs[i]
                    else:
                        iv_ssa = f"%iv_block{i}"
                    
                    # Compute offset: iv * block_size
                    offset_ssa = self.builder.fresh("offset")
                    self.builder.emit(
                        f'{offset_ssa} = arith.muli {iv_ssa}, {idx_ssa} : index'
                    )
                    offsets_ssa.append(offset_ssa)
                    sizes_ssa.append(idx_ssa)
                    output_dim_sizes.append(None)  # Dynamic (block sizes are runtime)
                else:
                    # Generic index - treat as offset with unknown size
                    offsets_ssa.append(idx_ssa)
                    size_ssa = self.builder.fresh("size")
                    self.builder.emit(f'{size_ssa} = arith.constant 1 : index')
                    sizes_ssa.append(size_ssa)
                    output_dim_sizes.append(1)  # Static size 1
            elif isinstance(idx, int):
                # Integer literal offset
                offset_ssa = self.builder.fresh("offset")
                self.builder.emit(f'{offset_ssa} = arith.constant {idx} : index')
                offsets_ssa.append(offset_ssa)
                size_ssa = self.builder.fresh("size")
                self.builder.emit(f'{size_ssa} = arith.constant 1 : index')
                sizes_ssa.append(size_ssa)
                output_dim_sizes.append(1)  # Static size 1
            else:
                # Fallback
                offset_ssa = self.builder.fresh("offset")
                self.builder.emit(f'{offset_ssa} = arith.constant 0 : index')
                offsets_ssa.append(offset_ssa)
                size_ssa = self.builder.fresh("size")
                self.builder.emit(f'{size_ssa} = arith.constant 1 : index')
                sizes_ssa.append(size_ssa)
                output_dim_sizes.append(1)  # Static size 1
        
        result = self.builder.fresh("slice")
        
        # Construct result type using tracked dimension sizes
        # Static dimensions use the concrete value, dynamic use '?'
        if 'tensor<' in tensor_type and '>' in tensor_type:
            # Extract dtype from input type
            content = tensor_type[tensor_type.find('<')+1 : tensor_type.rfind('>')]
            if 'x' in content:
                dtype_str = content.split('x')[-1]
            else:
                dtype_str = content
            
            # Build output dimensions
            output_dims = []
            for dim_size in output_dim_sizes:
                if dim_size is not None:
                    output_dims.append(str(dim_size))
                else:
                    output_dims.append("?")
            
            # Construct result type
            if output_dims:
                result_type = f"tensor<{'x'.join(output_dims)}x{dtype_str}>"
            else:
                result_type = f"tensor<{dtype_str}>"
        else:
            raise ValueError(f"Invalid tensor type: {tensor_type}")
            
        self.ctx.node_types[node.name] = result_type
        
        # Format as tensor.extract_slice
        # For static dimensions, use the concrete value in sizes bracket (not SSA)
        # For dynamic dimensions, use SSA values
        sizes_formatted = []
        for i, dim_size in enumerate(output_dim_sizes):
            if dim_size is not None:
                # Static dimension - use integer literal
                sizes_formatted.append(str(dim_size))
            else:
                # Dynamic dimension - use SSA value
                sizes_formatted.append(sizes_ssa[i])
        
        offsets_str = ", ".join(offsets_ssa)
        sizes_str = ", ".join(sizes_formatted)
        strides_str = ", ".join(["1"] * len(indices))
        
        # Use tensor.extract_slice syntax
        self.builder.emit(
            f'{result} = tensor.extract_slice {tensor_ssa}[{offsets_str}][{sizes_str}][{strides_str}] : '
            f'{tensor_type} to {result_type}'
        )
        
        self.ctx.node_values[node.name] = result
        return result
    
    def visit_store(self, node: fx.Node) -> str:
        """Generate tensor.insert_slice for tile storing.
        
        Replaces the custom helion.store with standard MLIR tensor.insert_slice.
        The tile sizes come from the value node's FakeTensor metadata.
        """
        from helion._compiler.variable_origin import BlockSizeOrigin
        
        tensor_node = node.args[0]
        indices = node.args[1]
        value = node.args[2]
        
        tensor_ssa = self.ctx.node_values.get(tensor_node.name, f"%{tensor_node.name}")
        tensor_type = self._get_tensor_type(tensor_node)  # Use dynamic type for destination
        
        value_ssa = self.ctx.node_values.get(value.name, f"%{value.name}") if isinstance(value, fx.Node) else str(value)
        value_type = self._get_tensor_type(value) if isinstance(value, fx.Node) else f"tensor<?x?xf32>"
        
        # Get FakeTensor from value node to determine tile sizes
        value_fake_tensor = value.meta.get("val") if isinstance(value, fx.Node) else None
        host_function = self.ctx.bound_kernel.host_function
        shape_env = self.ctx.bound_kernel.env.shape_env
        
        # Helper: resolve a SymInt to SSA using Origin lookup
        def resolve_symint_to_ssa(sym_int, dim_hint: int) -> str:
            """Resolve a SymInt to its SSA value using Origin-based lookup."""
            sym = sym_int._sympy_()
            origin_info = host_function.expr_to_origin.get(sym)
            origin = origin_info.origin if origin_info else None
            
            # BlockSizeOrigin -> use pre-emitted block_size_ssa
            if isinstance(origin, BlockSizeOrigin):
                block_id = origin.block_id
                ssa = self.ctx.block_size_ssa.get(block_id)
                if ssa is not None:
                    return ssa
            
            # Other origins -> use shape_env for concrete value
            if sym in shape_env.var_to_val:
                concrete_val = int(shape_env.var_to_val[sym])
                ssa = self.builder.fresh(f"size_dim{dim_hint}")
                self.builder.emit(f'{ssa} = arith.constant {concrete_val} : index')
                return ssa
            
            raise ValueError(f"Cannot resolve SymInt {sym} for dimension {dim_hint}")
        
        # Collect offset and size SSA values for each dimension
        # Track which dimensions have static sizes (for size bracket formatting)
        offsets_ssa = []
        sizes_ssa = []
        output_dim_sizes = []  # Store (static_size | None) for each dim
        
        for i, idx in enumerate(indices):
            if isinstance(idx, slice):
                # Full dimension slice - offset=0, size comes from source tensor dim
                zero_ssa = self.builder.fresh("zero")
                self.builder.emit(f'{zero_ssa} = arith.constant 0 : index')
                offsets_ssa.append(zero_ssa)
                
                # Check if value tensor's dimension is static
                src_dim_static = None
                if 'tensor<' in value_type:
                    type_content = value_type[value_type.find('<')+1 : value_type.rfind('>')]
                    if 'x' in type_content:
                        src_dims = type_content.split('x')[:-1]
                        if i < len(src_dims):
                            if src_dims[i] != '?':
                                try:
                                    src_dim_static = int(src_dims[i])
                                except ValueError:
                                    pass
                
                # Get dimension size from value tensor
                dim_ssa = self.builder.fresh("dim")
                dim_idx_ssa = self.builder.fresh("dim_idx")
                self.builder.emit(f'{dim_idx_ssa} = arith.constant {i} : index')
                self.builder.emit(f'{dim_ssa} = tensor.dim {value_ssa}, {dim_idx_ssa} : {value_type}')
                sizes_ssa.append(dim_ssa)
                output_dim_sizes.append(src_dim_static)
            elif isinstance(idx, fx.Node):
                idx_ssa = self.ctx.node_values.get(idx.name, f"%{idx.name}")
                
                if 'sym_size' in idx.name:
                    # Tile index as offset
                    offsets_ssa.append(idx_ssa)
                    
                    # Size from the value FakeTensor's shape
                    if value_fake_tensor is not None and i < value_fake_tensor.ndim:
                        dim_size = value_fake_tensor.size(i)
                        if hasattr(dim_size, '_sympy_'):
                            size_ssa = resolve_symint_to_ssa(dim_size, i)
                            sizes_ssa.append(size_ssa)
                            output_dim_sizes.append(None)  # Dynamic
                        else:
                            size_ssa = self.builder.fresh(f"size_dim{i}")
                            self.builder.emit(f'{size_ssa} = arith.constant {int(dim_size)} : index')
                            sizes_ssa.append(size_ssa)
                            output_dim_sizes.append(int(dim_size))  # Static
                    else:
                        sizes_ssa.append(idx_ssa)
                        output_dim_sizes.append(None)
                        
                elif 'block_size' in idx.name:
                    # block_size indicates the tile size, compute offset
                    if hasattr(self, 'current_loop_ivs') and i < len(self.current_loop_ivs):
                        iv_ssa = self.current_loop_ivs[i]
                    else:
                        iv_ssa = f"%iv_block{i}"
                    
                    offset_ssa = self.builder.fresh("offset")
                    self.builder.emit(f'{offset_ssa} = arith.muli {iv_ssa}, {idx_ssa} : index')
                    offsets_ssa.append(offset_ssa)
                    sizes_ssa.append(idx_ssa)
                    output_dim_sizes.append(None)  # Dynamic (block sizes are runtime)
                else:
                    offsets_ssa.append(idx_ssa)
                    size_ssa = self.builder.fresh("size")
                    self.builder.emit(f'{size_ssa} = arith.constant 1 : index')
                    sizes_ssa.append(size_ssa)
                    output_dim_sizes.append(1)  # Static size 1
            elif isinstance(idx, int):
                offset_ssa = self.builder.fresh("offset")
                self.builder.emit(f'{offset_ssa} = arith.constant {idx} : index')
                offsets_ssa.append(offset_ssa)
                size_ssa = self.builder.fresh("size")
                self.builder.emit(f'{size_ssa} = arith.constant 1 : index')
                sizes_ssa.append(size_ssa)
                output_dim_sizes.append(1)  # Static size 1
            else:
                offset_ssa = self.builder.fresh("offset")
                self.builder.emit(f'{offset_ssa} = arith.constant 0 : index')
                offsets_ssa.append(offset_ssa)
                size_ssa = self.builder.fresh("size")
                self.builder.emit(f'{size_ssa} = arith.constant 1 : index')
                sizes_ssa.append(size_ssa)
                output_dim_sizes.append(1)  # Static size 1
        
        result = self.builder.fresh("inserted")
        
        # Format as tensor.insert_slice
        # For static dimensions, use the concrete value in sizes bracket (not SSA)
        # For dynamic dimensions, use SSA values
        sizes_formatted = []
        for i, dim_size in enumerate(output_dim_sizes):
            if dim_size is not None:
                # Static dimension - use integer literal
                sizes_formatted.append(str(dim_size))
            else:
                # Dynamic dimension - use SSA value
                sizes_formatted.append(sizes_ssa[i])
        
        offsets_str = ", ".join(offsets_ssa)
        sizes_str = ", ".join(sizes_formatted)
        strides_str = ", ".join(["1"] * len(indices))
        
        # tensor.insert_slice %source into %dest[offsets][sizes][strides] : source_type into dest_type
        self.builder.emit(
            f'{result} = tensor.insert_slice {value_ssa} into {tensor_ssa}[{offsets_str}][{sizes_str}][{strides_str}] : '
            f'{value_type} into {tensor_type}'
        )
        
        self.ctx.node_values[node.name] = result
        return result
    
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
        
        current_ssa = self.ctx.node_values.get(tensor_node.name, f"%{tensor_node.name}") if isinstance(tensor_node, fx.Node) else str(tensor_node)
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
            
            for idx in slice_indices:
                if isinstance(idx, slice):
                    # Handle slice(None) aka [:]
                    # Offset 0
                    zero_ssa = self.builder.fresh("zero")
                    self.builder.emit(f'{zero_ssa} = arith.constant 0 : index')
                    offsets_ssa.append(zero_ssa)
                    
                    # Size: dim(input, input_dim_idx)
                    dim_ssa = self.builder.fresh("dim")
                    dim_idx_ssa = self.builder.fresh("dim_idx")
                    self.builder.emit(f'{dim_idx_ssa} = arith.constant {input_dim_idx} : index')
                    self.builder.emit(f'{dim_ssa} = tensor.dim {current_ssa}, {dim_idx_ssa} : {source_type}')
                    
                    sizes_ssa.append(dim_ssa)
                    strides_ssa.append("1") # Default stride 1
                    input_dim_idx += 1
                    
                elif isinstance(idx, int) or isinstance(idx, fx.Node):
                    # Integer indexing: offset=idx, size=1, stride=1
                    # Rank reducing -> we will drop this dimension in the result type
                    
                    if isinstance(idx, fx.Node):
                        idx_val = self.ctx.node_values.get(idx.name, f"%{idx.name}")
                    else:
                        idx_c = self.builder.fresh("idx")
                        self.builder.emit(f'{idx_c} = arith.constant {idx} : index')
                        idx_val = idx_c
                        
                    offsets_ssa.append(idx_val)
                    
                    one_ssa = self.builder.fresh("one")
                    self.builder.emit(f'{one_ssa} = arith.constant 1 : index')
                    sizes_ssa.append(one_ssa)
                    strides_ssa.append("1")
                    
                    input_dim_idx += 1
            
            # Count result dims to build type string
            res_rank = 0
            for idx in slice_indices:
                if isinstance(idx, slice):
                    res_rank += 1
            
            dims_str = "x".join(["?"] * res_rank)
            result_type = f"tensor<{dims_str}xf32>"
            extracted_type = result_type
            
            result_slice = self.builder.fresh("slice")
            
            offsets_str = ", ".join(offsets_ssa)
            sizes_str = ", ".join(sizes_ssa)
            strides_str = ", ".join(strides_ssa)
            
            self.builder.emit(
                f'{result_slice} = tensor.extract_slice {current_ssa}[{offsets_str}][{sizes_str}][{strides_str}] : '
                f'{source_type} to {result_type}'
            )
            
            extracted_ssa = result_slice
        
        # -----------------------------------------------------------
        # Step 2: Handle New Axis (tensor.expand_shape)
        # -----------------------------------------------------------
        
        has_newaxis = any(idx is None for idx in indices)
        
        if has_newaxis:
            # Reassociation logic
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
            dims_str = "x".join(result_dims)
            result_type = f"tensor<{dims_str}xf32>"
            
            result_expand = self.builder.fresh("expand")
            
            # Compute output_shape for dynamic dimensions
            # For each output dimension, we need its size as an SSA value
            output_shape_ssas = []
            extracted_dim_idx = 0  # Track which dimension of extracted_ssa we're on
            for i, dtype in enumerate(output_dim_types):
                if dtype == 'new':
                    # New dimension from None - always size 1
                    one_ssa = self.builder.fresh("one")
                    self.builder.emit(f'{one_ssa} = arith.constant 1 : index')
                    output_shape_ssas.append(one_ssa)
                else:
                    # Real dimension - get from source tensor
                    dim_idx_ssa = self.builder.fresh("dim_idx")
                    dim_ssa = self.builder.fresh("dim")
                    self.builder.emit(f'{dim_idx_ssa} = arith.constant {extracted_dim_idx} : index')
                    self.builder.emit(f'{dim_ssa} = tensor.dim {extracted_ssa}, {dim_idx_ssa} : {extracted_type}')
                    output_shape_ssas.append(dim_ssa)
                    extracted_dim_idx += 1
            
            output_shape_str = "[" + ", ".join(output_shape_ssas) + "]"
            
            self.builder.emit(
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
        The output dialect is controlled by ctx.aten_output_type:
        - "raw": torch dialect (torch.aten.*)
        - "linalg-on-tensors": linalg dialect on tensors
        - "tosa": TOSA dialect
        - "stablehlo": StableHLO dialect
        """
        target = node.target
        op_name, overload = get_aten_op_info(target)
        
        # Resolve all operands to SSA values
        operand_ssas = []
        for arg in node.args:
            if isinstance(arg, fx.Node):
                operand_ssas.append(self.ctx.node_values.get(arg.name, f"%{arg.name}"))
            elif arg is not None:
                operand_ssas.append(str(arg))
        
        # -------------------------------------------------------------------------
        # Use torch-mlir to generate MLIR for all ATen operations
        # The output type is controlled by ctx.aten_output_type
        # -------------------------------------------------------------------------
        
        # Get output type from context (defaults to "raw" for torch dialect)
        output_type = getattr(self.ctx, 'aten_output_type', 'raw')
        
        # Use torch-mlir to generate MLIR for this operation
        mlir_text = import_aten_node_to_mlir(node, output_type=output_type)
        if mlir_text is None:
             raise RuntimeError(f"Failed to lower ATen op: {node.name} ({target})")

        
        # Collect SSA values for tensor operands (matching what import_aten_node_to_mlir expects as args)
        tensor_operands = []
        def collect_operands(arg):
            if isinstance(arg, fx.Node):
                tensor_operands.append(self.ctx.node_values.get(arg.name, f"%{arg.name}"))
            return arg
            
        fx.map_arg(node.args, collect_operands)
        fx.map_arg(node.kwargs, collect_operands)
        
        # Inline the generated MLIR
        from .torch_mlir_helper import inline_torch_mlir_output
        result = inline_torch_mlir_output(mlir_text, tensor_operands, self.builder)
        
        self.ctx.node_values[node.name] = result
        return result
    
    def visit_unknown(self, node: fx.Node) -> str | None:
        """Handle unknown node targets."""
        # Log or emit a placeholder
        target_str = str(node.target)
        ssa = self.builder.fresh(f"unknown_{node.name}")
        
        attrs = format_attr_dict({
            "target": format_string_attr(target_str),
            "fx_node": format_string_attr(node.name),
        })
        
        self.builder.emit(
            f'{ssa} = "helion.unknown"(){attrs} : () -> index'
        )
        
        self.ctx.node_values[node.name] = ssa
        return ssa
    
    def _get_tensor_type(self, tensor_node: fx.Node | str) -> str:
        """Get MLIR tensor type for a tensor node.
        
        Lookup order:
        1. ctx.node_types (already computed for this node)
        2. ctx.host_tensor_types (pre-computed for _host_tensor nodes)
        3. ctx.arg_mlir_types (kernel arguments)
        4. Compute from FakeTensor if available
        5. Fallback to dynamic type
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
        
        # Fallback to dynamic type (should rarely reach here)
        return f"tensor<?x?xf32>"
