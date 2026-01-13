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
- aten.* compute -> helion.call_torch
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import torch
import torch.fx as fx
from torch.ops import aten

import helion.language.memory_ops as hl_memory_ops
import helion.language._tracing_ops as hl_tracing_ops
import helion.language.creation_ops as hl_creation_ops

from .mlir_builder import (
    format_attr_dict,
    format_string_attr,
    format_indices_attr,
    torch_dtype_to_mlir_element_type,
    format_tensor_type,
)

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
        
        # getitem -> map to loop result
        if target is getattr(__builtins__, 'getitem', None) or \
           (hasattr(target, '__name__') and target.__name__ == 'getitem'):
            return self.visit_getitem(node)
        
        # Check for operator.getitem
        import operator
        if target is operator.getitem:
            return self.visit_getitem(node)
        
        # aten.* compute operations -> helion.call_torch
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
                    self.current_loop_result = self.node_values.get(arg.name)
                    return self.current_loop_result
            # Multiple outputs
            results = []
            for arg in args:
                if isinstance(arg, fx.Node):
                    results.append(self.node_values.get(arg.name))
            self.current_loop_result = results[0] if results else None
            return self.current_loop_result
        
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
        
        # Resolve shape to SSA values
        shape_ssa = []
        for s in shape_nodes:
            if isinstance(s, fx.Node):
                shape_ssa.append(self.node_values.get(s.name, f"%{s.name}"))
            else:
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
        
        # Resolve iter_args
        iter_args_info = []
        for i, a in enumerate(args):
            if isinstance(a, fx.Node):
                ssa = self.node_values.get(a.name, f"%{a.name}")
                iter_args_info.append((f"acc_iter{i}", ssa, a.name))
            else:
                iter_args_info.append((f"acc_iter{i}", str(a), None))
        
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
        
        self.builder.emit(
            f'{result} = affine.for {iv} = 0 to {trip_count_ssa} '
            f'iter_args({iter_args_str}) -> ({result_types}) {{'
        )
        self.builder.push()
        
        # Set up iter_args inside loop - map placeholder names to iter SSA values
        old_loop_iter_args = self.loop_iter_args.copy()
        
        # Get node_args from ForLoopGraphInfo if available
        node_args = getattr(for_graph, 'node_args', [])
        
        # Map placeholders: the first placeholder after arg0_1 corresponds to iter_args
        # In ForLoopGraphInfo, placeholders are: arg0_1, then iter_args in order
        for i, (iter_name, _, orig_name) in enumerate(iter_args_info):
            # The placeholder will be accessed via visit_placeholder
            placeholder_name = f"arg{i + 1}_1" if i == 0 else f"arg{i}_1"
            self.loop_iter_args[placeholder_name] = f"%{iter_name}"
            
            # Also map the original iter arg name pattern
            self.loop_iter_args[f"arg0_{i + 2}"] = f"%{iter_name}"
        
        # For the first placeholder in loops (typically the iter_arg from outer)
        self.loop_iter_args["arg0_1"] = f"%{iter_args_info[0][0]}" if iter_args_info else "%acc_iter"
        
        self.loop_depth += 1
        
        # Visit inner graph
        self.visit_graph(for_graph)
        
        self.loop_depth -= 1
        
        # Restore loop_iter_args
        self.loop_iter_args = old_loop_iter_args
        
        # Emit yield with result from inner graph
        yield_value = self.current_loop_result or f"%{iter_args_info[0][0]}"
        self.builder.emit(f'affine.yield {yield_value} : {tensor_type}')
        
        self.builder.pop()
        self.builder.emit("}")
        
        self.node_values[node.name] = result
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
            ssa = f"%{tensor_name}"  # Default naming
        
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
        """Map getitem to the corresponding loop result."""
        source = node.args[0]
        index = node.args[1]
        
        # For _for_loop results, getitem extracts the i-th return value
        source_ssa = self.node_values.get(source.name, f"%{source.name}") if isinstance(source, fx.Node) else str(source)
        
        # For single-return loops, just use the result directly
        self.node_values[node.name] = source_ssa
        return source_ssa
    
    def visit_aten_compute(self, node: fx.Node) -> str:
        """Generate helion.call_torch for ATen compute ops."""
        target = node.target
        op_name = str(target).replace("aten.", "").replace(".default", "")
        
        # Resolve all operands
        operands = []
        for arg in node.args:
            if isinstance(arg, fx.Node):
                operands.append(self.node_values.get(arg.name, f"%{arg.name}"))
            elif arg is not None:
                operands.append(str(arg))
        
        result = self.builder.fresh(node.name)
        tensor_type = self.ctx.tensor_type
        
        attrs = format_attr_dict({
            "fn_name": format_string_attr(f"aten.{op_name}"),
            "fx_node": format_string_attr(node.name),
        })
        
        operand_str = ", ".join(operands)
        type_str = ", ".join([tensor_type] * len(operands))
        
        self.builder.emit(
            f'{result} = "helion.call_torch"({operand_str}){attrs} '
            f': ({type_str}) -> {tensor_type}'
        )
        
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
