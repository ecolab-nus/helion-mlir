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
        
    def register_graph(self, graph_id: int, graph_info: "GraphInfo") -> None:
        """Register a graph for later visitation (e.g., ForLoopGraphInfo)."""
        self.ctx.graphs[graph_id] = graph_info
    
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
        
        # Register type from kernel args if available
        for arg in self.ctx.kernel_args:
            if arg.name == node.name and arg.mlir_type:
                self.ctx.node_types[node.name] = arg.mlir_type
                break
                
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
        """Generate loom.get_symbol for symnode references."""
        name = node.args[0]  # e.g., 'block_size_0'
        ssa = self.builder.fresh(name.replace(".", "_"))
        
        # Emit: %ssa = "loom.get_symbol"() {name = "block_size_0"} : () -> index
        attrs = format_attr_dict({"name": format_string_attr(name)})
        self.builder.emit(
            f'{ssa} = "loom.get_symbol"(){attrs} : () -> index'
        )
        
        # Store in symbol table
        self.ctx.symbols[name] = ssa
        self.ctx.node_values[node.name] = ssa
        return ssa
    
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
        
        # Generate dynamic tensor type with correct number of dimensions
        dim_wildcards = "x".join(["?"] * len(shape_ssa))
        tensor_type = f"tensor<{dim_wildcards}x{dtype_str}>"
        
        # Step 1: Emit tensor.empty
        empty_ssa = self.builder.fresh("empty")
        shape_str = ", ".join(shape_ssa)
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
        
        # Generate dynamic tensor type with correct number of dimensions
        dim_wildcards = "x".join(["?"] * len(shape_ssa))
        tensor_type = f"tensor<{dim_wildcards}x{dtype_str}>"
        
        # Step 1: Emit tensor.empty
        empty_ssa = self.builder.fresh("empty")
        shape_str = ", ".join(shape_ssa)
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
        for (name, ssa, _), type_str in zip(iter_args_info, iter_args_types):
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
        
        # Visit inner graph
        self.visit_graph(for_graph)
        
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
        """Generate helion.phi for control flow merge."""
        lhs = node.args[0]
        rhs = node.args[1]
        
        # lhs is the initial value (before the loop), rhs is the loop result
        # Use the tracked initial accumulator for lhs
        if isinstance(lhs, fx.Node):
            lhs_ssa = self.ctx.initial_acc_ssa.get(lhs.name)
            if lhs_ssa is None:
                lhs_ssa = self.ctx.node_values.get(lhs.name, f"%{lhs.name}")
        else:
            lhs_ssa = str(lhs)
        
        # rhs is the loop result (getitem on _for_loop)
        rhs_ssa = self.ctx.node_values.get(rhs.name, f"%{rhs.name}") if isinstance(rhs, fx.Node) else str(rhs)
        
        ssa = self.builder.fresh("phi")
        
        # Infer type from LHS
        tensor_type = self._get_tensor_type(lhs) if isinstance(lhs, fx.Node) else self.ctx.tensor_type
        
        attrs = format_attr_dict({"fx_node": format_string_attr(node.name)})
        
        self.builder.emit(
            f'{ssa} = "helion.phi"({lhs_ssa}, {rhs_ssa}){attrs} '
            f': ({tensor_type}, {tensor_type}) -> {tensor_type}'
        )
        
        self.ctx.node_values[node.name] = ssa
        self.ctx.node_types[node.name] = tensor_type
        return ssa
    
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
        The extract_slice operation takes offsets, sizes, and strides to
        extract a sub-tensor from the source.
        
        For tiled access like x[tile_m, tile_k]:
        - offsets come from loop IVs multiplied by tile sizes
        - sizes come from the tile dimensions (block sizes)
        - strides are 1 for contiguous access
        """
        tensor_node = node.args[0]
        indices = node.args[1]  # [sym_size_int, block_size_2] or [block_size_0, block_size_1, slice(...)]
        
        tensor_ssa = self.ctx.node_values.get(tensor_node.name, f"%{tensor_node.name}")
        tensor_type = self._get_tensor_type(tensor_node)
        
        # Collect offset and size SSA values for each dimension
        offsets_ssa = []
        sizes_ssa = []
        
        for i, idx in enumerate(indices):
            if isinstance(idx, slice):
                # Full dimension slice - offset=0, size comes from tensor dim
                zero_ssa = self.builder.fresh("zero")
                self.builder.emit(f'{zero_ssa} = arith.constant 0 : index')
                offsets_ssa.append(zero_ssa)
                
                # Get dimension size from tensor
                dim_ssa = self.builder.fresh("dim")
                dim_idx_ssa = self.builder.fresh("dim_idx")
                self.builder.emit(f'{dim_idx_ssa} = arith.constant {i} : index')
                self.builder.emit(f'{dim_ssa} = tensor.dim {tensor_ssa}, {dim_idx_ssa} : {tensor_type}')
                sizes_ssa.append(dim_ssa)
            elif isinstance(idx, fx.Node):
                # Get the SSA value for this index
                idx_ssa = self.ctx.node_values.get(idx.name, f"%{idx.name}")
                
                # Determine if this is a loop IV (sym_size_int) or a block size
                if 'sym_size' in idx.name:
                    # sym_size_int represents the tile index within the iteration
                    # The offset is: loop_iv * block_size
                    # For now, use loop IV directly as it's already scaled
                    offsets_ssa.append(idx_ssa)
                    
                    # Size needs to come from block_size - look for corresponding block_size
                    # For now, emit a symbolic size lookup
                    size_ssa = self.builder.fresh("tile_size")
                    self.builder.emit(f'{size_ssa} = "loom.get_symbol"() {{name = "tile_size_{i}"}} : () -> index')
                    sizes_ssa.append(size_ssa)
                elif 'block_size' in idx.name:
                    # block_size node indicates the tile size for this dimension
                    # The offset comes from the corresponding loop IV
                    # Compute offset: iv * block_size
                    
                    # Get the current loop IV
                    if hasattr(self, 'current_loop_ivs') and i < len(self.current_loop_ivs):
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
                else:
                    # Generic index - treat as offset with unknown size
                    offsets_ssa.append(idx_ssa)
                    size_ssa = self.builder.fresh("size")
                    self.builder.emit(f'{size_ssa} = arith.constant 1 : index')
                    sizes_ssa.append(size_ssa)
            elif isinstance(idx, int):
                # Integer literal offset
                offset_ssa = self.builder.fresh("offset")
                self.builder.emit(f'{offset_ssa} = arith.constant {idx} : index')
                offsets_ssa.append(offset_ssa)
                size_ssa = self.builder.fresh("size")
                self.builder.emit(f'{size_ssa} = arith.constant 1 : index')
                sizes_ssa.append(size_ssa)
            else:
                # Fallback
                offset_ssa = self.builder.fresh("offset")
                self.builder.emit(f'{offset_ssa} = arith.constant 0 : index')
                offsets_ssa.append(offset_ssa)
                size_ssa = self.builder.fresh("size")
                self.builder.emit(f'{size_ssa} = arith.constant 1 : index')
                sizes_ssa.append(size_ssa)
        
        result = self.builder.fresh("slice")
        
        # Infer result type - same rank as source tensor, same element type
        # format: tensor<?x?xf32>
        if 'tensor<' in tensor_type and '>' in tensor_type:
            # Extract content between <>
            content = tensor_type[tensor_type.find('<')+1 : tensor_type.rfind('>')]
            if 'x' in content:
                parts = content.split('x')
                dtype_str = parts[-1]
                rank = len(parts) - 1
            else:
                # 0-D or 1-D with no dims?
                rank = 0
                dtype_str = content
            
            # Construct dynamic type with same rank
            if rank > 0:
                dim_wildcards = "x".join(["?"] * rank)
                result_type = f"tensor<{dim_wildcards}x{dtype_str}>"
            else:
                result_type = f"tensor<{dtype_str}>"
        else:
            # Fallback
            result_type = self.ctx.tensor_type
            
        self.ctx.node_types[node.name] = result_type
        
        # Format as tensor.extract_slice
        rank = len(indices)
        offsets_str = ", ".join(offsets_ssa)
        sizes_str = ", ".join(sizes_ssa)
        strides_str = ", ".join(["1"] * rank)
        
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
        The insert_slice operation inserts a tile back into the destination tensor.
        
        For tiled store like out[tile_m, tile_n] = acc:
        - offsets come from loop IVs multiplied by tile sizes
        - sizes come from the tile dimensions
        - strides are 1 for contiguous access
        """
        tensor_node = node.args[0]
        indices = node.args[1]
        value = node.args[2]
        
        tensor_ssa = self.ctx.node_values.get(tensor_node.name, f"%{tensor_node.name}")
        tensor_type = self._get_tensor_type(tensor_node)  # Use dynamic type for destination
        
        value_ssa = self.ctx.node_values.get(value.name, f"%{value.name}") if isinstance(value, fx.Node) else str(value)
        value_type = self._get_tensor_type(value) if isinstance(value, fx.Node) else f"tensor<?x?xf32>"
        
        # Collect offset and size SSA values for each dimension
        offsets_ssa = []
        sizes_ssa = []
        
        for i, idx in enumerate(indices):
            if isinstance(idx, slice):
                # Full dimension slice - offset=0, size comes from source tensor dim
                zero_ssa = self.builder.fresh("zero")
                self.builder.emit(f'{zero_ssa} = arith.constant 0 : index')
                offsets_ssa.append(zero_ssa)
                
                # Get dimension size from value tensor
                dim_ssa = self.builder.fresh("dim")
                dim_idx_ssa = self.builder.fresh("dim_idx")
                self.builder.emit(f'{dim_idx_ssa} = arith.constant {i} : index')
                self.builder.emit(f'{dim_ssa} = tensor.dim {value_ssa}, {dim_idx_ssa} : {value_type}')
                sizes_ssa.append(dim_ssa)
            elif isinstance(idx, fx.Node):
                idx_ssa = self.ctx.node_values.get(idx.name, f"%{idx.name}")
                
                if 'sym_size' in idx.name:
                    # Tile index as offset
                    offsets_ssa.append(idx_ssa)
                    # Size from a symbol lookup
                    size_ssa = self.builder.fresh("tile_size")
                    self.builder.emit(f'{size_ssa} = "loom.get_symbol"() {{name = "tile_size_{i}"}} : () -> index')
                    sizes_ssa.append(size_ssa)
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
                else:
                    offsets_ssa.append(idx_ssa)
                    size_ssa = self.builder.fresh("size")
                    self.builder.emit(f'{size_ssa} = arith.constant 1 : index')
                    sizes_ssa.append(size_ssa)
            elif isinstance(idx, int):
                offset_ssa = self.builder.fresh("offset")
                self.builder.emit(f'{offset_ssa} = arith.constant {idx} : index')
                offsets_ssa.append(offset_ssa)
                size_ssa = self.builder.fresh("size")
                self.builder.emit(f'{size_ssa} = arith.constant 1 : index')
                sizes_ssa.append(size_ssa)
            else:
                offset_ssa = self.builder.fresh("offset")
                self.builder.emit(f'{offset_ssa} = arith.constant 0 : index')
                offsets_ssa.append(offset_ssa)
                size_ssa = self.builder.fresh("size")
                self.builder.emit(f'{size_ssa} = arith.constant 1 : index')
                sizes_ssa.append(size_ssa)
        
        result = self.builder.fresh("inserted")
        
        # Format as tensor.insert_slice
        rank = len(indices)
        offsets_str = ", ".join(offsets_ssa)
        sizes_str = ", ".join(sizes_ssa)
        strides_str = ", ".join(["1"] * rank)
        
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
        if hasattr(self, '_loop_result_values') and isinstance(source, fx.Node):
            if source.name in self.ctx.loop_result_values:
                # Multi-value loop result - use #index syntax
                result_ssa = f"{source_ssa}#{index}"
                self.ctx.node_values[node.name] = result_ssa
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
            
            # Result type
            res_rank = len(output_dim_types)
            dims_str = "x".join(["?"] * res_rank)
            result_type = f"tensor<{dims_str}xf32>"
            
            result_expand = self.builder.fresh("expand")
            
            self.builder.emit(
                f'{result_expand} = tensor.expand_shape {extracted_ssa} {reassoc_str} : '
                f'{extracted_type} into {result_type}'
            )
            
            self.ctx.node_values[node.name] = result_expand
            return result_expand
            
        else:
            self.ctx.node_values[node.name] = extracted_ssa
            return extracted_ssa
    
    def visit_aten_compute(self, node: fx.Node) -> str:
        """Generate MLIR for ATen compute ops using torch-mlir.
        
        Uses torch-mlir's FxImporter to generate proper MLIR for ATen operations.
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
        # Use torch-mlir to generate torch dialect MLIR for all ATen operations
        # -------------------------------------------------------------------------
        
        # Use torch-mlir to generate MLIR for this operation
        mlir_text = import_aten_node_to_mlir(node)
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
        """Get MLIR tensor type for a tensor node."""
        if isinstance(tensor_node, fx.Node):
            name = tensor_node.name
        else:
            name = str(tensor_node)
        
        # Look up in registered node types
        if name in self.ctx.node_types:
            return self.ctx.node_types[name]
            
        # Look up in kernel args
        for arg in self.ctx.kernel_args:
            if arg.name == name and arg.mlir_type:
                return arg.mlir_type
        
        # Look up in loop iter args (if it's a placeholder in a loop)
        if name in self.loop_iter_args:
            # We might not have the type easily available for loop args unless we track them
            # For now, try to find the corresponding acc_iter info if possible,
            # or rely on the caller to handle this case.
            pass

        # Fallback to f32 if we can't determine type (temporary for migration)
        # But generally we should raise an error if type is unknown
        return f"tensor<?x?xf32>"
