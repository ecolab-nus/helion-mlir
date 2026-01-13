"""Torch-MLIR integration helper using FxImporter infrastructure.

This module provides utilities to convert ATen operations from Helion Device IR
into MLIR using torch-mlir's FxImporter. It supports generating either:
- Raw torch dialect MLIR
- Linalg-on-tensors MLIR (via automatic lowering)

Key Functions:
- import_aten_node: Import a single FX node using torch-mlir
- TorchMLIRNodeImporter: Class for importing FX nodes to MLIR text
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.fx as fx
from torch._ops import OpOverload

if TYPE_CHECKING:
    from .lowering_context import LoweringContext


def get_aten_op_info(target: Any) -> tuple[str, str]:
    """Extract ATen operation name and overload from target.
    
    Args:
        target: FX node target (typically torch._ops.OpOverload)
        
    Returns:
        Tuple of (op_name, overload) e.g., ("addmm", "default")
    """
    if isinstance(target, OpOverload):
        return target.__name__, target._overloadname
    
    # Fallback: parse from string representation
    target_str = str(target)
    if "aten." in target_str:
        parts = target_str.replace("aten::", "aten.").split(".")
        if len(parts) >= 2:
            op_name = parts[1]
            overload = parts[2] if len(parts) > 2 else "default"
            return op_name, overload
    
    return target_str, "default"


class TorchMLIRNodeImporter:
    """Imports FX nodes to MLIR using torch-mlir's FxImporter.
    
    This class wraps torch-mlir's infrastructure to convert individual 
    ATen operations to MLIR text that can be embedded in helion MLIR.
    """
    
    def __init__(self, output_type: str = "linalg-on-tensors"):
        """Initialize the importer.
        
        Args:
            output_type: Target MLIR dialect - "raw" for torch dialect,
                        "linalg-on-tensors" for linalg, etc.
        """
        self.output_type = output_type
        self._context = None
        self._importer = None
        
    def _ensure_initialized(self):
        """Lazily initialize torch-mlir context and importer."""
        if self._context is not None:
            return
            
        try:
            from torch_mlir import ir
            from torch_mlir.dialects import torch as torch_d
            from torch_mlir.extras.fx_importer import FxImporter
            
            self._context = ir.Context()
            torch_d.register_dialect(self._context)
            self._importer = FxImporter(context=self._context)
        except ImportError as e:
            raise RuntimeError(
                f"torch-mlir not available: {e}. "
                "Please install torch-mlir to use this functionality."
            )
    
    def import_graph(
        self,
        graph: fx.Graph,
        func_name: str = "aten_op",
    ) -> str:
        """Import an FX graph to MLIR.
        
        Args:
            graph: FX Graph to import
            func_name: Name for the generated MLIR function
            
        Returns:
            MLIR text representation
        """
        self._ensure_initialized()
        
        from torch_mlir.compiler_utils import OutputType, lower_mlir_module
        
        # Import the graph
        self._importer.import_stateless_graph(graph, func_name=func_name)
        
        # Get the module and lower if needed
        module = self._importer.module
        
        if self.output_type != "raw":
            module = lower_mlir_module(
                False,  # verbose
                OutputType.get(self.output_type),
                module
            )
        
        return str(module)
    
    def import_node(
        self,
        node: fx.Node,
        input_tensors: list[torch.Tensor],
    ) -> str:
        """Import a single FX node to MLIR by creating a minimal graph.
        
        Args:
            node: The FX node to import (should be an ATen op)
            input_tensors: List of tensor shapes/dtypes for inputs
            
        Returns:
            MLIR text for the operation
        """
        self._ensure_initialized()
        
        # Create a minimal FX graph containing just this node
        # This requires wrapping it in a proper graph structure
        graph = fx.Graph()
        
        # Create placeholder nodes for inputs
        placeholder_nodes = []
        fake_tensor_iter = iter(input_tensors)
        
        # We need to map args while maintaining structure (lists, tuples)
        def map_arg(arg):
            if isinstance(arg, fx.Node):
                try:
                    val = next(fake_tensor_iter)
                    placeholder = graph.placeholder(f"input_{len(placeholder_nodes)}")
                    placeholder.meta["val"] = val
                    placeholder_nodes.append(placeholder)
                    return placeholder
                except StopIteration:
                    raise RuntimeError("Mismatch between node args and input fake tensors")
            return arg

        new_args = fx.map_arg(node.args, map_arg)
        new_kwargs = fx.map_arg(node.kwargs, map_arg)
        
        # Create the operation node
        op_node = graph.call_function(node.target, args=new_args, kwargs=new_kwargs)
        
        # Try to infer output type from node metadata
        if "val" in node.meta:
            val = node.meta["val"]
            if isinstance(val, torch.Tensor) and not isinstance(val, torch._subclasses.fake_tensor.FakeTensor):
                 from torch._subclasses.fake_tensor import FakeTensorMode
                 with FakeTensorMode():
                     val = torch.empty(val.shape, dtype=val.dtype, device="meta")
            op_node.meta["val"] = val
        
        # Create output
        graph.output(op_node)
        
        # Import and return
        return self.import_graph(graph, func_name="aten_op")



def create_fake_tensors_for_node(node: fx.Node) -> list[torch.Tensor]:
    """Create fake tensors for a node's inputs based on metadata.
    
    Args:
        node: FX node whose inputs need fake tensors
        
    Returns:
        List of fake tensors matching input shapes/dtypes (flattened)
    """
    from torch._subclasses.fake_tensor import FakeTensorMode
    
    fake_mode = FakeTensorMode()
    fake_tensors = []
    
    def process_arg(arg):
        if isinstance(arg, fx.Node):
            # Try to get existing val
            val = arg.meta.get("val")
            if isinstance(val, torch.Tensor):
                fake = torch.empty(val.shape, dtype=val.dtype, device="meta")
                fake_tensors.append(fake)
            elif isinstance(val, (int, float, bool)):
                # For scalars, append the value itself
                fake_tensors.append(val)
            else:
                # Default fallback
                fake = torch.empty([1, 1], dtype=torch.float32, device="meta")
                fake_tensors.append(fake)
        return arg

    with fake_mode:
        fx.map_arg(node.args, process_arg)
        fx.map_arg(node.kwargs, process_arg)
            
    return fake_tensors



# Global cache for context logic
_CONTEXT_CACHE = None

class ContextCache:
    def __init__(self):
        from torch_mlir import ir
        from torch_mlir.dialects import torch as torch_d
        self.context = ir.Context()
        torch_d.register_dialect(self.context)

def get_cached_context():
    global _CONTEXT_CACHE
    if _CONTEXT_CACHE is None:
        _CONTEXT_CACHE = ContextCache()
    return _CONTEXT_CACHE.context

def import_aten_node_to_mlir(
    node: fx.Node,
    output_type: str = "linalg-on-tensors",
) -> Optional[str]:
    """Import an ATen FX node to MLIR using torch-mlir.
    
    This is the main entry point for converting ATen operations.
    
    Args:
        node: FX node containing an ATen operation
        output_type: Target MLIR dialect ("raw", "linalg-on-tensors", "tosa", "stablehlo")
        
    Returns:
        MLIR text for the operation, or None if import fails
    """
    try:
        # Create a fresh importer but use the cached context
        import torch_mlir
        # We need to manually construct importer with cached context to avoid default init
        # But TorchMLIRNodeImporter logic is:
        # _ensure_initialized() creates context.
        # We should modify TorchMLIRNodeImporter to accept context or subclass it.
        # Or just manually use FxImporter here since we essentially rewrote the logic anyway.
        
        # Let's use TorchMLIRNodeImporter but inject the context
        importer = TorchMLIRNodeImporter(output_type=output_type)
        importer._context = get_cached_context()
        from torch_mlir.extras.fx_importer import FxImporter
        importer._importer = FxImporter(context=importer._context)
        
        fake_tensors = create_fake_tensors_for_node(node)
        return importer.import_node(node, fake_tensors)
        return importer.import_node(node, fake_tensors)
    except Exception as e:
        # Log the error but don't fail
        import traceback
        traceback.print_exc()
        import warnings
        warnings.warn(f"Failed to import ATen node {node.name}: {e}")
        return None



import re

def inline_torch_mlir_output(
    mlir_text: str, 
    operands: list[str], 
    builder
) -> str:
    """Inline torch-mlir generated text into the current builder.
    
    Args:
        mlir_text: The full MLIR module text from torch-mlir.
        operands: SSA values to use as arguments.
        builder: The MLIR builder to emit to.
        
    Returns:
        The SSA value of the result.
    """
    lines = mlir_text.splitlines()
    ssa_map = {}
    
    # 1. Map function arguments
    # Torch-mlir output always starts args with %arg0, %arg1...
    for i, op in enumerate(operands):
        ssa_map[f"%arg{i}"] = op
        
    # Regex to identify SSAs
    ssa_pattern = re.compile(r'%([a-zA-Z0-9_]+)')
    
    # helper to replace SSAs in a string
    def replace_ssas(text, mapping):
        def repl(m):
            name = m.group(1)
            full = f"%{name}"
            return mapping.get(full, full)
        return ssa_pattern.sub(repl, text)

    # Find body start
    start_idx = 0
    for i, line in enumerate(lines):
        if "func.func @aten_op" in line:
            start_idx = i + 1
            break
            
    result_ssa = None
    
    for line in lines[start_idx:]:
        line = line.strip()
        if not line or line.startswith("}"):
            continue
            
        # Handle Return
        if line.startswith("return"):
            parts = line.split()
            if len(parts) >= 2:
                # "return %res : type"
                ret = parts[1]
                result_ssa = ssa_map.get(ret, ret)
            continue
            
        # Handle Block Args (e.g. ^bb0(%a: f32, %b: f32):)
        if line.startswith("^"):
            # This is a block label with args. Args are definitions.
            # format: ^bb0(%arg: type, ...):
            # We need to extract the arg names and map them to fresh names.
            
            # Find definitions
            # split by ( and )
            pre, rest = line.split("(", 1)
            args_part, post = rest.rsplit(")", 1)
            
            # Split args
            args_list = args_part.split(",")
            new_args_list = []
            for arg_def in args_list:
                # "%name: type"
                arg_def = arg_def.strip()
                if ":" in arg_def:
                    name_part, type_part = arg_def.split(":", 1)
                    name = name_part.strip()
                    if name.startswith("%"):
                        fresh = builder.fresh("blk_arg")
                        ssa_map[name] = fresh
                        new_args_list.append(f"{fresh}: {type_part}")
                    else:
                        new_args_list.append(arg_def)
                else:
                    new_args_list.append(arg_def)
            
            new_line = f"{pre}({', '.join(new_args_list)}){post}"
            builder.emit(new_line)
            continue

        # Handle Assignment
        if "=" in line:
            lhs, rhs = line.split("=", 1)
            lhs = lhs.strip()
            rhs = rhs.strip()
            
            # LHS are definitions
            lhs_vars = [x.strip() for x in lhs.split(",")]
            new_lhs_vars = []
            for v in lhs_vars:
                if v.startswith("%"):
                    fresh = builder.fresh("t")
                    ssa_map[v] = fresh
                    new_lhs_vars.append(fresh)
                else:
                    new_lhs_vars.append(v)
            
            new_lhs = ", ".join(new_lhs_vars)
            
            # RHS are usages
            new_rhs = replace_ssas(rhs, ssa_map)
            
            builder.emit(f"{new_lhs} = {new_rhs}")
            continue
            
        # Handle standalone ops (like linalg.yield)
        new_line = replace_ssas(line, ssa_map)
        builder.emit(new_line)

    return result_ssa

