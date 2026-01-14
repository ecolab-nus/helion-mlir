"""IR Visitor for walking Device IR graphs and generating MLIR.

This module implements a visitor pattern for converting Helion Device IR
to MLIR by walking FX graph nodes instruction-by-instruction.

The visitor dispatches to specific handlers based on the node's target:
- _get_symnode -> loom.get_symbol
- full -> helion.full
- _for_loop -> affine.for + recursive visit
- _phi -> helion.phi
- _host_tensor -> function argument mapping
- aten.sym_size.int -> inline concrete value
- load/store -> helion.load/helion.store
- aten.* compute -> torch.aten.* (via torch-mlir dialect)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

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
    format_indices_attr,
    torch_dtype_to_mlir_element_type,
    format_tensor_type,
)
from .torch_mlir_helper import (
    TorchMLIRNodeImporter,
    get_aten_op_info,
    import_aten_node_to_mlir,
)

# Import lowerings to trigger registration of ATen op lowerings
from . import lowerings  # noqa: F401 - triggers @register_lowering decorators

if TYPE_CHECKING:
    from helion._compiler.device_ir import GraphInfo, ForLoopGraphInfo, RootGraphInfo
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
        
        # FX node name -> MLIR SSA value
        self.node_values: dict[str, str] = {}
        
        # Graph ID -> GraphInfo for nested graphs (ForLoopGraphInfo)
        self.graphs: dict[int, "GraphInfo"] = {}
        
        # Current loop context for iter_args management
        self.loop_iter_args: dict[str, str] = {}  # placeholder name -> SSA
        
        # Current loop result (set by inner graph output)
        self.current_loop_result: str | None = None
        
        # Depth tracking for nested loops
        self.loop_depth: int = 0
        
        # Track output tensor SSA and type
        self.output_tensor_ssa: str | None = None
        self.output_tensor_type: str | None = None
        
        # Track the initial accumulator (before loop) for phi
        self.initial_acc_ssa: dict[str, str] = {}  # acc node name -> initial SSA
        
        # Pre-computed block sizes and trip counts (set by generate_mlir)
        self.block_size_ssa: dict[int, str] = {}  # block_id -> SSA
        self.reduction_trip_counts: dict[int, str] = {}  # block_id -> trip count SSA
        
    def register_graph(self, graph_id: int, graph_info: "GraphInfo") -> None:
        """Register a graph for later visitation (e.g., ForLoopGraphInfo)."""
        self.graphs[graph_id] = graph_info
    
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
            self.node_values[node.name] = ssa
            return ssa
        
        # Otherwise it's a function argument placeholder
        # For now, just create a placeholder SSA value
        ssa = f"%{node.name}"
        self.node_values[node.name] = ssa
        return ssa
    
    def visit_call_function(self, node: fx.Node) -> str | None:
        """Dispatch call_function nodes to specific handlers."""
        target = node.target
        
        # _get_symnode -> loom.get_symbol
        if target is hl_tracing_ops._get_symnode:
            return self.visit_get_symnode(node)
        
        # full -> helion.full
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
        
        # load -> helion.load
        if target is hl_memory_ops.load:
            return self.visit_load(node)
        
        # store -> helion.store
        if target is hl_memory_ops.store:
            return self.visit_store(node)
        
        # _mask_to -> shortcircuit (pass through input, boundary check placeholder)
        if target is hl_tracing_ops._mask_to:
            return self.visit_mask_to(node)
        
        # subscript -> helion.subscript (view/indexing operation)
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
        
        # aten.full.default -> helion.full (handle specially, torch-mlir can't handle FX node shapes)
        if target is aten.full.default:
            return self.visit_aten_full(node)
        
        # aten.* compute operations
        # First check if there's a registered lowering in op_registry
        if hasattr(target, '__module__') and 'aten' in str(target):
            from .op_registry import LoweringRegistry
            if LoweringRegistry.has(target):
                # Sync node_values to ctx.fx_value_map for registered lowerings
                self.ctx.fx_value_map.update(self.node_values)
                # Use registered lowering (e.g., linalg.matmul for addmm)
                result = LoweringRegistry.emit_node(self.ctx, node)
                if result is not None:
                    self.node_values[node.name] = result
                    return result
            # Fall back to generic visit_aten_compute
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
                    result = self.node_values.get(arg.name)
                    self.current_loop_result = [result]  # Store as list for consistency
                    return result
            # Multiple outputs - store ALL results
            results = []
            for arg in args:
                if isinstance(arg, fx.Node):
                    results.append(self.node_values.get(arg.name))
            self.current_loop_result = results  # Store entire list
            return results[0] if results else None

        
        if isinstance(args, fx.Node):
            self.current_loop_result = self.node_values.get(args.name)
            return self.current_loop_result
        
        return None
    
    def visit_get_attr(self, node: fx.Node) -> str:
        """Handle get_attr nodes."""
        # Just create a placeholder SSA
        ssa = self.builder.fresh(node.name)
        self.node_values[node.name] = ssa
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
        self.node_values[node.name] = ssa
        return ssa
    
    def visit_full(self, node: fx.Node) -> str:
        """Generate helion.full for tensor initialization.
        
        For special values like -inf, we emit arith.constant with IEEE 754 hex 
        representation and pass it as an operand to helion.full.
        """
        import math
        
        shape_nodes = node.args[0]  # List of FX nodes or values
        fill_value = node.args[1] if len(node.args) > 1 else 0.0
        dtype = node.args[2] if len(node.args) > 2 else torch.float32
        
        # Resolve shape to SSA values - integer literals need to become arith.constant
        shape_ssa = []
        for s in shape_nodes:
            if isinstance(s, fx.Node):
                shape_ssa.append(self.node_values.get(s.name, f"%{s.name}"))
            elif isinstance(s, int):
                # Integer literals need to be emitted as arith.constant
                const_ssa = self.builder.fresh("dim")
                self.builder.emit(f'{const_ssa} = arith.constant {s} : index')
                shape_ssa.append(const_ssa)
            else:
                # Fallback for other values
                shape_ssa.append(str(s))

        
        ssa = self.builder.fresh("full")
        dtype_str = torch_dtype_to_mlir_element_type(dtype) if dtype else "f32"
        
        # Handle special float values like -inf, inf, nan
        fill_value_ssa = None
        fill_value_attr = None
        
        if isinstance(fill_value, float) and (math.isinf(fill_value) or math.isnan(fill_value)):
            # Emit arith.constant with IEEE 754 hex representation
            const_ssa = self.builder.fresh("fill_val")
            if math.isinf(fill_value):
                if fill_value < 0:
                    # -inf for f32 = 0xFF800000
                    hex_val = "0xFF800000"
                else:
                    # +inf for f32 = 0x7F800000
                    hex_val = "0x7F800000"
            else:
                # nan for f32 = 0x7FC00000
                hex_val = "0x7FC00000"
            
            self.builder.emit(f'{const_ssa} = arith.constant {hex_val} : {dtype_str}')
            fill_value_ssa = const_ssa
        else:
            fill_value_attr = fill_value
        
        # Build the op
        shape_str = ", ".join(shape_ssa)
        type_str = ", ".join(["index"] * len(shape_ssa))
        
        if fill_value_ssa:
            # Pass fill value as operand
            operands = f"{shape_str}, {fill_value_ssa}"
            operand_types = f"{type_str}, {dtype_str}"
            attrs = format_attr_dict({"dtype": dtype_str})
            self.builder.emit(
                f'{ssa} = "helion.full"({operands}){attrs} '
                f': ({operand_types}) -> tensor<?x?x{dtype_str}>'
            )
        else:
            # Use fill_value as attribute
            attrs = format_attr_dict({
                "fill_value": fill_value_attr,
                "dtype": dtype_str,
            })
            self.builder.emit(
                f'{ssa} = "helion.full"({shape_str}){attrs} '
                f': ({type_str}) -> tensor<?x?x{dtype_str}>'
            )
        
        self.node_values[node.name] = ssa
        
        # Track initial accumulator for phi
        self.initial_acc_ssa[node.name] = ssa
        
        return ssa
    
    def visit_aten_full(self, node: fx.Node) -> str:
        """Generate helion.full for aten.full.default (from torch.full_like).
        
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
                shape_ssa.append(self.node_values.get(s.name, f"%{s.name}"))
            elif isinstance(s, int):
                # Integer literals need to be emitted as arith.constant
                const_ssa = self.builder.fresh("dim")
                self.builder.emit(f'{const_ssa} = arith.constant {s} : index')
                shape_ssa.append(const_ssa)
            else:
                # Fallback for other values
                shape_ssa.append(str(s))

        
        ssa = self.builder.fresh("full")
        dtype_str = torch_dtype_to_mlir_element_type(dtype) if dtype else "f32"
        
        # Handle special float values like -inf, inf, nan
        fill_value_ssa = None
        fill_value_attr = None
        
        if isinstance(fill_value, float) and (math.isinf(fill_value) or math.isnan(fill_value)):
            # Emit arith.constant with IEEE 754 hex representation
            const_ssa = self.builder.fresh("fill_val")
            if math.isinf(fill_value):
                if fill_value < 0:
                    # -inf for f32 = 0xFF800000
                    hex_val = "0xFF800000"
                else:
                    # +inf for f32 = 0x7F800000
                    hex_val = "0x7F800000"
            else:
                # nan for f32 = 0x7FC00000
                hex_val = "0x7FC00000"
            
            self.builder.emit(f'{const_ssa} = arith.constant {hex_val} : {dtype_str}')
            fill_value_ssa = const_ssa
        else:
            fill_value_attr = fill_value
        
        # Build the op - determine tensor type based on shape dimensions
        shape_str = ", ".join(shape_ssa)
        type_str = ", ".join(["index"] * len(shape_ssa))
        
        # Generate dynamic tensor type with correct number of dimensions
        dim_wildcards = "x".join(["?"] * len(shape_ssa))
        tensor_type = f"tensor<{dim_wildcards}x{dtype_str}>"
        
        if fill_value_ssa:
            # Pass fill value as operand
            operands = f"{shape_str}, {fill_value_ssa}"
            operand_types = f"{type_str}, {dtype_str}"
            attrs = format_attr_dict({"dtype": dtype_str})
            self.builder.emit(
                f'{ssa} = "helion.full"({operands}){attrs} '
                f': ({operand_types}) -> {tensor_type}'
            )
        else:
            # Use fill_value as attribute
            attrs = format_attr_dict({
                "fill_value": fill_value_attr,
                "dtype": dtype_str,
            })
            self.builder.emit(
                f'{ssa} = "helion.full"({shape_str}){attrs} '
                f': ({type_str}) -> {tensor_type}'
            )
        
        self.node_values[node.name] = ssa
        
        # Track initial accumulator for phi
        self.initial_acc_ssa[node.name] = ssa
        
        return ssa
    
    def visit_for_loop(self, node: fx.Node) -> str:

        """Generate affine.for and visit the inner ForLoopGraphInfo."""
        graph_id = node.args[0]
        begin = node.args[1]  # [0]
        end = node.args[2]    # [x_size1]
        args = node.args[3]   # [acc]
        
        # Get the ForLoopGraphInfo
        for_graph = self.graphs.get(graph_id)
        if for_graph is None:
            raise ValueError(f"ForLoopGraphInfo with graph_id={graph_id} not registered")
        
        # Get block_ids from the graph
        block_ids = getattr(for_graph, 'block_ids', [self.loop_depth])
        block_id = block_ids[0] if block_ids else 2
        
        # Use pre-computed trip count (computed outside affine.parallel)
        trip_count_ssa = self.reduction_trip_counts.get(block_id)
        if trip_count_ssa is None:
            # Fallback: find any reduction trip count
            if self.reduction_trip_counts:
                trip_count_ssa = list(self.reduction_trip_counts.values())[0]
            else:
                # Last resort: emit trip count here (will cause affine symbol error)
                trip_count_ssa = "%unknown_trip_count"
        
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
                ssa = self.node_values.get(a.name, f"%{a.name}")
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
        
        # Determine tensor type for iter_args
        tensor_type = self.ctx.tensor_type

        
        # Emit affine.for
        iv = f"%iv_block{block_id}"
        result = self.builder.fresh(f"for_result_{graph_id}")
        
        # Build iter_args string
        iter_args_parts = []
        for name, ssa, _ in iter_args_info:
            iter_args_parts.append(f"%{name} = {ssa}")
        iter_args_str = ", ".join(iter_args_parts)
        
        # Result types
        result_types = ", ".join([tensor_type] * len(iter_args_info))
        
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
        for i, (_, ssa, orig_name) in enumerate(readonly_args_info):
            placeholder_name = f"arg{i}_1"
            self.loop_iter_args[placeholder_name] = ssa  # Use original SSA
        
        # Map placeholders for loop-carried iter_args
        num_readonly = len(readonly_args_info)
        for i, (iter_name, _, orig_name) in enumerate(iter_args_info):
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
            yield_types = ", ".join([tensor_type] * len(self.current_loop_result))
            self.builder.emit(f'affine.yield {yield_values} : {yield_types}')
        else:
            # Single yield value (backward compatible)
            yield_value = self.current_loop_result[0] if isinstance(self.current_loop_result, list) else (self.current_loop_result or f"%{iter_args_info[0][0]}")
            self.builder.emit(f'affine.yield {yield_value} : {tensor_type}')
        
        self.builder.pop()
        self.builder.emit("}")
        
        # Store the loop result - for multi-value loops, this SSA represents all results
        # Individual results are extracted via getitem
        self.node_values[node.name] = result
        
        # Also store the result SSAs for each output index
        # This allows visit_getitem to extract individual results
        if isinstance(self.current_loop_result, list) and len(self.current_loop_result) > 1:
            self._loop_result_values = {
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
            lhs_ssa = self.initial_acc_ssa.get(lhs.name)
            if lhs_ssa is None:
                lhs_ssa = self.node_values.get(lhs.name, f"%{lhs.name}")
        else:
            lhs_ssa = str(lhs)
        
        # rhs is the loop result (getitem on _for_loop)
        rhs_ssa = self.node_values.get(rhs.name, f"%{rhs.name}") if isinstance(rhs, fx.Node) else str(rhs)
        
        ssa = self.builder.fresh("phi")
        tensor_type = self.ctx.tensor_type
        
        attrs = format_attr_dict({"fx_node": format_string_attr(node.name)})
        
        self.builder.emit(
            f'{ssa} = "helion.phi"({lhs_ssa}, {rhs_ssa}){attrs} '
            f': ({tensor_type}, {tensor_type}) -> {tensor_type}'
        )
        
        self.node_values[node.name] = ssa
        return ssa
    
    def visit_new_var(self, node: fx.Node) -> str:
        """Pass through _new_var nodes (just forward the input value)."""
        arg = node.args[0]
        if isinstance(arg, fx.Node):
            ssa = self.node_values.get(arg.name, f"%{arg.name}")
        else:
            ssa = str(arg)
        
        self.node_values[node.name] = ssa
        return ssa
    
    def visit_host_tensor(self, node: fx.Node) -> str:
        """Map host tensor reference to function argument or allocate output."""
        tensor_name = node.args[0]  # 'x', 'y', 'out'
        
        # Check if this is the output tensor
        if tensor_name == 'out':
            # Allocate output tensor
            if self.output_tensor_ssa is None:
                # Find first input tensor for alloc_like
                first_input_ssa = None
                first_input_type = None
                for arg in self.ctx.kernel_args:
                    if arg.is_tensor and arg.name != 'out':
                        first_input_ssa = f"%{arg.name}"
                        first_input_type = arg.mlir_type or self.ctx.tensor_type
                        break
                
                # Determine output shape from loop extents
                from .mlir_builder import format_shape_attr
                output_shape = [loop.total_extent for loop in self.ctx.outer_loops]
                while len(output_shape) < 2:
                    output_shape.append(None)
                shape_attr = format_shape_attr(output_shape)
                
                # Determine output type
                self.output_tensor_type = format_tensor_type(output_shape, self.ctx.element_type)
                self.output_tensor_ssa = self.builder.fresh("out")
                
                if first_input_ssa:
                    attrs = format_attr_dict({"shape": shape_attr})
                    self.builder.emit(
                        f'{self.output_tensor_ssa} = "helion.alloc_like"({first_input_ssa}){attrs} '
                        f': ({first_input_type}) -> {self.output_tensor_type}'
                    )
                else:
                    # Fallback: emit alloc without template
                    attrs = format_attr_dict({"shape": shape_attr})
                    self.builder.emit(
                        f'{self.output_tensor_ssa} = "helion.alloc"(){attrs} '
                        f': () -> {self.output_tensor_type}'
                    )
            
            self.node_values[node.name] = self.output_tensor_ssa
            self.ctx.host_tensors[tensor_name] = self.output_tensor_ssa
            return self.output_tensor_ssa
        
        # Look up in pre-registered host tensors
        ssa = self.ctx.host_tensors.get(tensor_name)
        if ssa is None:
            # This is a derived/view tensor (like q_view, k_view) - emit host_ref
            ssa = self.builder.fresh(tensor_name)
            attrs = format_attr_dict({
                "name": format_string_attr(tensor_name),
                "fx_node": format_string_attr(node.name),
            })
            self.builder.emit(
                f'{ssa} = "helion.host_ref"(){attrs} '
                f': () -> {self.ctx.tensor_type}'
            )
            # Register for reuse
            self.ctx.host_tensors[tensor_name] = ssa
        
        self.node_values[node.name] = ssa
        return ssa

    
    def visit_sym_size(self, node: fx.Node) -> str:
        """Handle aten.sym_size.int - maps to outer loop tile index.
        
        In ForLoopGraphInfo, sym_size_int is used to get the tile index
        from the outer loops. For example:
        - sym_size_int(arg0_1, 0) -> %iv_block0 (first outer loop IV)
        - sym_size_int(arg0_1, 1) -> %iv_block1 (second outer loop IV)
        """
        tensor_node = node.args[0]
        dim = node.args[1]
        
        # Map dimension to outer loop IV
        # dim 0 -> block 0 -> %iv_block0
        # dim 1 -> block 1 -> %iv_block1
        if dim < len(self.ctx.outer_loops):
            block_id = self.ctx.outer_loops[dim].block_id
            iv_name = f"%iv_block{block_id}"
            self.node_values[node.name] = iv_name
            return iv_name
        
        # Fallback for reduction dimension or other cases
        # Look up if this corresponds to a reduction loop
        if dim < len(self.ctx.outer_loops) + len(self.ctx.reduction_loops):
            idx = dim - len(self.ctx.outer_loops)
            if idx < len(self.ctx.reduction_loops):
                block_id = self.ctx.reduction_loops[idx].block_id
                iv_name = f"%iv_block{block_id}"
                self.node_values[node.name] = iv_name
                return iv_name
        
        # Last fallback: just use the dim as block id
        iv_name = f"%iv_block{dim}"
        self.node_values[node.name] = iv_name
        return iv_name
    
    def visit_load(self, node: fx.Node) -> str:
        """Generate helion.load with affine expressions."""
        tensor_node = node.args[0]
        indices = node.args[1]  # [sym_size_int, block_size_2]
        
        tensor_ssa = self.node_values.get(tensor_node.name, f"%{tensor_node.name}")
        tensor_type = self._get_tensor_type(tensor_node)
        
        # Build indices
        indices_ssa = []
        for idx in indices:
            if isinstance(idx, fx.Node):
                indices_ssa.append(self.node_values.get(idx.name, f"%{idx.name}"))
            else:
                indices_ssa.append(str(idx))
        
        result = self.builder.fresh("load")
        result_type = self.ctx.tensor_type
        
        # Format indices as attribute
        indices_attr = format_indices_attr(indices_ssa)
        
        attrs = format_attr_dict({
            "indices": indices_attr,
            "fx_node": format_string_attr(node.name),
        })
        
        self.builder.emit(
            f'{result} = "helion.load"({tensor_ssa}){attrs} '
            f': ({tensor_type}) -> {result_type}'
        )
        
        self.node_values[node.name] = result
        return result
    
    def visit_store(self, node: fx.Node) -> str:
        """Generate helion.store with affine expressions."""
        tensor_node = node.args[0]
        indices = node.args[1]
        value = node.args[2]
        
        tensor_ssa = self.node_values.get(tensor_node.name, f"%{tensor_node.name}")
        
        # Use output tensor type if this is the output
        if self.output_tensor_type:
            tensor_type = self.output_tensor_type
        else:
            tensor_type = self._get_tensor_type(tensor_node)
        
        value_ssa = self.node_values.get(value.name, f"%{value.name}") if isinstance(value, fx.Node) else str(value)
        value_type = self.ctx.tensor_type
        
        # Build indices
        indices_ssa = []
        for idx in indices:
            if isinstance(idx, fx.Node):
                indices_ssa.append(self.node_values.get(idx.name, f"%{idx.name}"))
            else:
                indices_ssa.append(str(idx))
        
        indices_attr = format_indices_attr(indices_ssa)
        
        attrs = format_attr_dict({
            "indices": indices_attr,
            "fx_node": format_string_attr(node.name),
        })
        
        self.builder.emit(
            f'"helion.store"({tensor_ssa}, {value_ssa}){attrs} '
            f': ({tensor_type}, {value_type}) -> ()'
        )
        
        self.node_values[node.name] = tensor_ssa  # Store doesn't produce a new value
        return tensor_ssa
    
    def visit_getitem(self, node: fx.Node) -> str:
        """Map getitem to the corresponding loop result.
        
        For multi-value affine.for loops, getitem(_for_loop, i) extracts
        the i-th return value as %result#i syntax.
        """
        source = node.args[0]
        index = node.args[1]
        
        # For _for_loop results, getitem extracts the i-th return value
        source_ssa = self.node_values.get(source.name, f"%{source.name}") if isinstance(source, fx.Node) else str(source)
        
        # Check if this is extracting from a multi-value loop result
        if hasattr(self, '_loop_result_values') and isinstance(source, fx.Node):
            if source.name in self._loop_result_values:
                # Multi-value loop result - use #index syntax
                result_ssa = f"{source_ssa}#{index}"
                self.node_values[node.name] = result_ssa
                return result_ssa
        
        # For single-return loops, just use the result directly
        self.node_values[node.name] = source_ssa
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
            ssa = self.node_values.get(tensor_node.name, f"%{tensor_node.name}")
        else:
            ssa = str(tensor_node)
        
        self.node_values[node.name] = ssa
        return ssa
    
    def visit_subscript(self, node: fx.Node) -> str:
        """Generate helion.subscript for tensor view/indexing operations.
        
        subscript is used for tensor slicing and newaxis operations.
        For now, we emit a placeholder helion.subscript op that returns
        a tensor type. Proper indexing support would need to handle
        the slice objects properly.
        
        TODO: Implement proper slice/index handling.
        """
        tensor_node = node.args[0]
        indices = node.args[1] if len(node.args) > 1 else []
        
        tensor_ssa = self.node_values.get(tensor_node.name, f"%{tensor_node.name}") if isinstance(tensor_node, fx.Node) else str(tensor_node)
        tensor_type = self.ctx.tensor_type
        
        result = self.builder.fresh("subscript")
        
        # Format indices as string attribute (simplified - ignores slice details)
        indices_str = str(indices)
        
        attrs = format_attr_dict({
            "indices": format_string_attr(indices_str),
            "fx_node": format_string_attr(node.name),
        })
        
        self.builder.emit(
            f'{result} = "helion.subscript"({tensor_ssa}){attrs} '
            f': ({tensor_type}) -> {tensor_type}'
        )
        
        self.node_values[node.name] = result
        return result
    
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
                operand_ssas.append(self.node_values.get(arg.name, f"%{arg.name}"))
            elif arg is not None:
                operand_ssas.append(str(arg))
        
        tensor_type = self.ctx.tensor_type
        
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
                tensor_operands.append(self.node_values.get(arg.name, f"%{arg.name}"))
            return arg
            
        fx.map_arg(node.args, collect_operands)
        fx.map_arg(node.kwargs, collect_operands)
        
        # Inline the generated MLIR
        from .torch_mlir_helper import inline_torch_mlir_output
        result = inline_torch_mlir_output(mlir_text, tensor_operands, self.builder)
        
        self.node_values[node.name] = result
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
        
        self.node_values[node.name] = ssa
        return ssa
    
    def _get_tensor_type(self, tensor_node: fx.Node | str) -> str:
        """Get MLIR tensor type for a tensor node."""
        if isinstance(tensor_node, fx.Node):
            name = tensor_node.name
        else:
            name = str(tensor_node)
        
        # Look up in kernel args
        for arg in self.ctx.kernel_args:
            if arg.name == name and arg.mlir_type:
                return arg.mlir_type
        
        # Default to dynamic tensor type
        return self.ctx.tensor_type
