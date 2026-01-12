"""MLIR lowerings for Helion tracing operations.

This module provides lowerings for internal tracing operations:
- helion.language._tracing_ops._host_tensor -> tensor annotations
- helion.language._tracing_ops._get_symnode -> symbolic index values
- helion.language._tracing_ops._new_var -> value copy/rename
- helion.language._tracing_ops._constant_tensor -> constant values
- helion.language._tracing_ops._and, _or, _not -> logical operations
- helion.language._tracing_ops._mask_to -> masked value operations
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import helion.language._tracing_ops as hl_tracing_ops

from ..op_registry import register_lowering
from ..mlir_builder import format_attr_dict, format_string_attr
from .base import MLIRLowering, PassthroughLowering

if TYPE_CHECKING:
    import torch.fx
    from ..lowering_context import LoweringContext


@register_lowering(hl_tracing_ops._host_tensor)
class HostTensorLowering(MLIRLowering):
    """Lowering for _host_tensor -> tensor annotation.
    
    Host tensors are tensors that originate from the host and are passed
    to the kernel as arguments, or intermediate tensors derived from them.
    This lowering maps them to function arguments or emits reference ops.
    """
    
    def emit(self, ctx: "LoweringContext", node: "torch.fx.Node") -> str | None:
        """Map a host tensor to a function argument or emit a reference.
        
        The _host_tensor op carries a debug name that indicates which
        tensor it corresponds to.
        """
        builder = ctx.builder
        
        # Get the debug name from the argument
        debug_name = str(node.args[0]) if node.args else ""
        
        # First, try to find a matching kernel argument by name
        tensor_arg = ctx.get_tensor_arg_by_name(debug_name)
        if tensor_arg and tensor_arg.ssa_name:
            ssa_value = tensor_arg.ssa_name
            ctx.fx_value_map[node.name] = ssa_value
            return ssa_value
        
        # Check if this is an output tensor reference
        if "out" in debug_name.lower() or "result" in debug_name.lower():
            ssa_value = ctx.out_value or "%out0"
            ctx.fx_value_map[node.name] = ssa_value
            return ssa_value
        
        # For intermediate/derived tensors (e.g., k_view, q_view),
        # emit a helion.host_ref operation that references the tensor by name
        # This allows the MLIR to track which host tensor is being accessed
        ssa_value = builder.fresh(debug_name.replace(".", "_").replace("-", "_"))
        
        # Get tensor type from node metadata if available
        tensor_type = self._get_tensor_type_from_node(ctx, node)
        
        attrs = format_attr_dict({
            "name": format_string_attr(debug_name),
            "fx_node": format_string_attr(node.name),
        })
        
        builder.emit(
            f'{ssa_value} = "helion.host_ref"(){attrs} : () -> {tensor_type}'
        )
        
        ctx.fx_value_map[node.name] = ssa_value
        return ssa_value
    
    def _get_tensor_type_from_node(self, ctx: "LoweringContext", node: "torch.fx.Node") -> str:
        """Get MLIR tensor type from FX node metadata."""
        import torch
        from ..mlir_builder import format_tensor_type, torch_dtype_to_mlir_element_type
        
        val = node.meta.get("val")
        if isinstance(val, torch.Tensor):
            shape = [int(s) if not isinstance(s, torch.SymInt) else None for s in val.shape]
            element_type = torch_dtype_to_mlir_element_type(val.dtype)
            return format_tensor_type(shape, element_type)
        
        # Fallback to default tensor type
        return ctx.tensor_type


@register_lowering(hl_tracing_ops._get_symnode)
class GetSymnodeLowering(MLIRLowering):
    """Lowering for _get_symnode -> symbolic index value.
    
    Symnode operations represent symbolic integer values like dimensions
    or tile sizes. They map to index values in MLIR.
    """
    
    def emit(self, ctx: "LoweringContext", node: "torch.fx.Node") -> str | None:
        """Map a symnode to an MLIR index value.
        
        The _get_symnode op carries a debug name that can help identify
        which symbolic value it represents.
        """
        builder = ctx.builder
        
        # Get the debug name
        debug_name = node.args[0] if node.args else "symnode"
        
        # Try to get the actual value from metadata
        val = node.meta.get("val")
        
        if isinstance(val, int):
            # Concrete value - emit as constant
            result = builder.emit_index_constant(val)
        else:
            # Symbolic value - check if it corresponds to a known symbol
            name_str = str(debug_name).lower()
            
            if name_str in ctx.symbolic_arg_ssa:
                result = ctx.symbolic_arg_ssa[name_str]
            elif name_str in ctx.dims_map and ctx.dims_map[name_str]:
                result = ctx.dims_map[name_str]
            else:
                # Emit a placeholder - this might be resolved later
                builder.emit_comment(f"symnode: {node.name} = {debug_name}")
                result = builder.fresh(f"sym_{node.name}")
                builder.emit(f"{result} = arith.constant 0 : index  // placeholder for {debug_name}")
        
        ctx.fx_value_map[node.name] = result
        return result


@register_lowering(hl_tracing_ops._new_var)
class NewVarLowering(MLIRLowering):
    """Lowering for _new_var -> value copy/rename.
    
    The _new_var operation creates a copy of a value to ensure proper
    handling of phi nodes and value merging in control flow.
    """
    
    def emit(self, ctx: "LoweringContext", node: "torch.fx.Node") -> str | None:
        """Create a copy of the input value."""
        builder = ctx.builder
        
        if not node.args:
            return None
        
        input_arg = node.args[0]
        input_ssa = self._get_input_ssa(ctx, input_arg)
        
        # For tensors, we might want to just alias
        # For now, emit a comment and use the same value
        builder.emit_comment(f"new_var: {node.name} <- {input_ssa}")
        
        ctx.fx_value_map[node.name] = input_ssa
        return input_ssa
    
    def _get_input_ssa(self, ctx: "LoweringContext", arg: object) -> str:
        """Get SSA value for input."""
        import torch.fx
        
        if isinstance(arg, torch.fx.Node):
            if arg.name in ctx.fx_value_map:
                return ctx.fx_value_map[arg.name]
        return "%v0"


@register_lowering(hl_tracing_ops._constant_tensor)
class ConstantTensorLowering(MLIRLowering):
    """Lowering for _constant_tensor -> constant value.
    
    Constant tensors are scalar constants created inside the kernel.
    """
    
    def emit(self, ctx: "LoweringContext", node: "torch.fx.Node") -> str | None:
        """Emit a constant tensor value."""
        builder = ctx.builder
        
        # Get value and dtype from arguments
        value = node.args[0] if node.args else 0
        dtype_str = node.args[1] if len(node.args) > 1 else "f32"
        
        builder.emit_comment(f"constant_tensor: {node.name} = {value} ({dtype_str})")
        
        # Emit as a splat constant
        result = builder.fresh("const")
        builder.emit(
            f'{result} = "helion.constant"() {{value = {value} : {dtype_str}}} : () -> {ctx.tensor_type}'
        )
        
        ctx.fx_value_map[node.name] = result
        return result


@register_lowering(hl_tracing_ops._and)
class AndLowering(MLIRLowering):
    """Lowering for _and -> logical AND."""
    
    def emit(self, ctx: "LoweringContext", node: "torch.fx.Node") -> str | None:
        """Emit logical AND operation."""
        builder = ctx.builder
        
        if len(node.args) < 2:
            return None
        
        lhs = self._get_value(ctx, node.args[0])
        rhs = self._get_value(ctx, node.args[1])
        
        result = builder.fresh("and")
        builder.emit(f"{result} = arith.andi {lhs}, {rhs} : i1")
        
        ctx.fx_value_map[node.name] = result
        return result
    
    def _get_value(self, ctx: "LoweringContext", arg: object) -> str:
        import torch.fx
        if isinstance(arg, torch.fx.Node):
            return ctx.fx_value_map.get(arg.name, "%false")
        return "%false"


@register_lowering(hl_tracing_ops._or)
class OrLowering(MLIRLowering):
    """Lowering for _or -> logical OR."""
    
    def emit(self, ctx: "LoweringContext", node: "torch.fx.Node") -> str | None:
        """Emit logical OR operation."""
        builder = ctx.builder
        
        if len(node.args) < 2:
            return None
        
        lhs = self._get_value(ctx, node.args[0])
        rhs = self._get_value(ctx, node.args[1])
        
        result = builder.fresh("or")
        builder.emit(f"{result} = arith.ori {lhs}, {rhs} : i1")
        
        ctx.fx_value_map[node.name] = result
        return result
    
    def _get_value(self, ctx: "LoweringContext", arg: object) -> str:
        import torch.fx
        if isinstance(arg, torch.fx.Node):
            return ctx.fx_value_map.get(arg.name, "%false")
        return "%false"


@register_lowering(hl_tracing_ops._not)
class NotLowering(MLIRLowering):
    """Lowering for _not -> logical NOT."""
    
    def emit(self, ctx: "LoweringContext", node: "torch.fx.Node") -> str | None:
        """Emit logical NOT operation."""
        builder = ctx.builder
        
        if not node.args:
            return None
        
        operand = self._get_value(ctx, node.args[0])
        
        # XOR with true to get NOT
        result = builder.fresh("not")
        true_const = builder.fresh("true")
        builder.emit(f"{true_const} = arith.constant true")
        builder.emit(f"{result} = arith.xori {operand}, {true_const} : i1")
        
        ctx.fx_value_map[node.name] = result
        return result
    
    def _get_value(self, ctx: "LoweringContext", arg: object) -> str:
        import torch.fx
        if isinstance(arg, torch.fx.Node):
            return ctx.fx_value_map.get(arg.name, "%false")
        return "%false"


@register_lowering(hl_tracing_ops._mask_to)
class MaskToLowering(MLIRLowering):
    """Lowering for _mask_to -> masked value operation.
    
    Sets masked-out values of a tile to a specific value, used for
    dot products and reductions.
    """
    
    def emit(self, ctx: "LoweringContext", node: "torch.fx.Node") -> str | None:
        """Emit mask_to operation."""
        builder = ctx.builder
        
        if len(node.args) < 2:
            return None
        
        tensor_arg = node.args[0]
        mask_value = node.args[1]
        
        tensor_ssa = self._get_tensor_ssa(ctx, tensor_arg)
        
        # Emit as a placeholder operation
        result = builder.fresh("masked")
        attrs = format_attr_dict({"mask_value": str(mask_value)})
        builder.emit(
            f'{result} = "helion.mask_to"({tensor_ssa}){attrs} '
            f": ({ctx.tensor_type}) -> {ctx.tensor_type}"
        )
        
        ctx.fx_value_map[node.name] = result
        return result
    
    def _get_tensor_ssa(self, ctx: "LoweringContext", arg: object) -> str:
        import torch.fx
        if isinstance(arg, torch.fx.Node):
            return ctx.fx_value_map.get(arg.name, "%tensor")
        return "%tensor"


@register_lowering(hl_tracing_ops._inductor_lowering_extra)
class InductorLoweringExtraLowering(PassthroughLowering):
    """Lowering for _inductor_lowering_extra -> passthrough.
    
    These are intermediate values from Inductor lowerings that don't
    need direct MLIR representation.
    """
    pass
