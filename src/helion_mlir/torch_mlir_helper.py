"""Torch-MLIR integration helper using FxImporter infrastructure.

This module provides utilities to convert ATen operations from Helion Device IR
into MLIR using torch-mlir's FxImporter. It supports generating either:
- Linalg-on-tensors MLIR (via automatic lowering)

Key Functions:
- import_aten_node: Import a single FX node using torch-mlir
- TorchMLIRNodeImporter: Class for importing FX nodes to MLIR text
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.fx as fx
from torch._ops import OpOverload

if TYPE_CHECKING:
    from .lowering_context import LoweringContext

class TorchMLIRNodeImporter:
    """Imports FX nodes to MLIR using torch-mlir's FxImporter.
    
    This class wraps torch-mlir's infrastructure to convert individual 
    ATen operations to MLIR text that can be embedded in helion MLIR.
    """
    
    def __init__(self):
        """Initialize the importer."""
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
        
        from torch_mlir.compiler_utils import (
            OutputType, 
            lower_mlir_module,
            run_pipeline_with_repro_report,
        )
        
        # Import the graph
        self._importer.import_stateless_graph(graph, func_name=func_name)
        
        # Get the module
        module = self._importer.module
        
        # Run the torch backend pipeline (RAW -> torch backend IR)
        run_pipeline_with_repro_report(
            module,
            "builtin.module(func.func(torch-match-quantized-custom-ops), torchdynamo-export-to-torch-backend-pipeline{ extra-library=})",
            "Lowering TorchFX IR -> Torch Backend IR",
        )
        
        # Lower to target dialect (torch backend IR -> linalg/tosa/stablehlo)
        module = lower_mlir_module(
            False,  # verbose
            OutputType.get("linalg-on-tensors"),
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
        
        from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor
        fake_mode = FakeTensorMode()
        
        def to_fake_tensor(val):
            """Convert a tensor or tuple of tensors to FakeTensor format."""
            if isinstance(val, torch.Tensor) and not isinstance(val, FakeTensor):
                with fake_mode:
                    return torch.empty(val.shape, dtype=val.dtype, device="meta")
            elif isinstance(val, (tuple, list)):
                # Handle tuple/list of tensors (e.g., max.dim returns (values, indices))
                converted = [to_fake_tensor(v) for v in val]
                return tuple(converted) if isinstance(val, tuple) else converted
            return val
        
        # We need to map args while maintaining structure (lists, tuples)
        def map_arg(arg):
            if isinstance(arg, fx.Node):
                try:
                    val = next(fake_tensor_iter)
                    placeholder = graph.placeholder(f"input_{len(placeholder_nodes)}")
                    # Ensure placeholder val is a FakeTensor
                    placeholder.meta["val"] = to_fake_tensor(val)
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
        # Handle both single tensors and tuple outputs (like max.dim)
        import operator
        
        if "val" in node.meta:
            val = node.meta["val"]
            fake_val = to_fake_tensor(val)
            op_node.meta["val"] = fake_val
            
            # Check if output is a tuple (like max.dim returns (values, indices))
            if isinstance(fake_val, tuple):
                # Decompose tuple output using getitem nodes
                # This is required because torch-mlir's _graph_to_function_meta
                # passes each result_node to node_val_to_type, which fails on tuples
                getitem_nodes = []
                for i, elem_val in enumerate(fake_val):
                    getitem_node = graph.call_function(operator.getitem, (op_node, i))
                    getitem_node.meta["val"] = elem_val
                    getitem_nodes.append(getitem_node)
                
                # Output the unpacked tuple elements
                graph.output(tuple(getitem_nodes))
            else:
                # Single output
                graph.output(op_node)
        else:
            # No val metadata, just output the node
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
                raise RuntimeError(f"Unsupported arg type: {type(arg)}")
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
) -> str:
    """Import an ATen FX node to MLIR using torch-mlir.
    
    This is the main entry point for converting ATen operations.
    
    Args:
        node: FX node containing an ATen operation
        
    Returns:
        MLIR text for the operation
        
    Raises:
        RuntimeError: If torch-mlir fails to import/lower the node
    """
    
    # Create a fresh importer but use the cached context
    importer = TorchMLIRNodeImporter()
    importer._context = get_cached_context()
    from torch_mlir.extras.fx_importer import FxImporter
    importer._importer = FxImporter(context=importer._context)
    
    fake_tensors = create_fake_tensors_for_node(node)
    return importer.import_node(node, fake_tensors)










import re

def inline_torch_mlir_output(
    mlir_text: str, 
    operands: list[str], 
    builder
) -> str:
    """Inline torch-mlir generated text into the current builder.
    
    Handles multiline operations like linalg.generic which have block bodies:
    ```
    %3 = linalg.generic {...} ins(...) outs(...) {
    ^bb0(%in: f32, ...):
      %4 = arith.addf %in, %in_0 : f32
      linalg.yield %4 : f32
    } -> tensor<32x64xf32>
    ```
    
    Args:
        mlir_text: The full MLIR module text from torch-mlir.
        operands: SSA values to use as arguments.
        builder: The MLIR builder to emit to.
        
    Returns:
        The SSA value of the result.
    """
    lines = mlir_text.splitlines()
    ssa_map = {}
    
    # Track affine map aliases to inline them
    affine_map_aliases = {}  # old_alias -> (new_alias, affine_map_def)
    affine_map_counter = getattr(builder, '_affine_map_counter', 0)
    
    # 1. First pass: collect affine map definitions
    # These are lines like: #map = affine_map<(d0, d1) -> (d0, d1)>
    affine_map_pattern = re.compile(r'^(#\w+)\s*=\s*(affine_map<.+>)\s*$')
    
    for line in lines:
        line_stripped = line.strip()
        match = affine_map_pattern.match(line_stripped)
        if match:
            old_alias = match.group(1)
            affine_map_def = match.group(2)
            new_alias = f"#map{affine_map_counter}"
            affine_map_counter += 1
            affine_map_aliases[old_alias] = (new_alias, affine_map_def)
    
    builder._affine_map_counter = affine_map_counter
    
    # Helper to replace affine map aliases with inline definitions
    # Use regex to ensure we match whole aliases (e.g., #map but not #map1)
    def replace_affine_maps(text):
        for old_alias, (new_alias, affine_map_def) in affine_map_aliases.items():
            # Escape the alias for regex and use word boundary to prevent partial matches
            # e.g., #map should not match inside #map1
            pattern = re.escape(old_alias) + r'(?![0-9a-zA-Z_])'
            text = re.sub(pattern, affine_map_def, text)
        return text
    
    # 2. Map function arguments
    for i, op in enumerate(operands):
        ssa_map[f"%arg{i}"] = op
        
    # Regex to identify SSAs
    ssa_pattern = re.compile(r'%([a-zA-Z0-9_][a-zA-Z0-9_.+-]*)')
    
    def replace_ssas(text, mapping):
        def repl(m):
            name = m.group(1)
            full = f"%{name}"
            return mapping.get(full, full)
        return ssa_pattern.sub(repl, text)

    # Find function body start and end
    func_start_idx = 0
    func_end_idx = len(lines)
    brace_count = 0
    
    for i, line in enumerate(lines):
        if "func.func @aten_op" in line:
            func_start_idx = i + 1
            brace_count = 1  # Opening brace of func
            break
    
    # Find the closing brace of the function
    for i in range(func_start_idx, len(lines)):
        line = lines[i]
        brace_count += line.count('{') - line.count('}')
        if brace_count == 0:
            func_end_idx = i
            break
            
    result_ssa = None
    
    # Process lines within the function body
    i = func_start_idx
    while i < func_end_idx:
        line = lines[i].strip()
        i += 1
        
        if not line:
            continue
        
        # Skip affine map definitions (already handled)
        if affine_map_pattern.match(line):
            continue
            
        # Handle Return - extract result SSA
        if line.startswith("return"):
            parts = line.split()
            if len(parts) >= 2:
                ret = parts[1].rstrip(',')
                result_ssa = ssa_map.get(ret, ret)
            continue
            
        # Handle lines that are just closing braces with result type: "} -> tensor<...>"
        if line.startswith("}"):
            # This is a closing brace for an inline region, emit it
            new_line = replace_ssas(line, ssa_map)
            new_line = replace_affine_maps(new_line)
            builder.emit(new_line)
            continue
            
        # Handle Block Args (e.g. ^bb0(%a: f32, %b: f32):)
        if line.startswith("^"):
            if "(" in line and ")" in line:
                pre, rest = line.split("(", 1)
                args_part, post = rest.rsplit(")", 1)
                
                args_list = args_part.split(",")
                new_args_list = []
                for arg_def in args_list:
                    arg_def = arg_def.strip()
                    if ":" in arg_def:
                        name_part, type_part = arg_def.split(":", 1)
                        name = name_part.strip()
                        if name.startswith("%"):
                            fresh = builder.fresh("blk_arg")
                            ssa_map[name] = fresh
                            new_args_list.append(f"{fresh}:{type_part}")
                        else:
                            new_args_list.append(arg_def)
                    else:
                        new_args_list.append(arg_def)
                
                new_line = f"{pre}({', '.join(new_args_list)}){post}"
            else:
                new_line = line
            new_line = replace_affine_maps(new_line)
            builder.emit(new_line)
            continue

        # Handle Assignment (lines with '=' that define SSA values)
        if "=" in line and not line.startswith("cf.assert"):
            # Check if the '=' is part of an SSA assignment (not inside attribute syntax like '=')
            # Simple heuristic: if line starts with '%', it's an SSA assignment
            first_eq = line.index("=")
            lhs = line[:first_eq].strip()
            rhs = line[first_eq+1:].strip()
            
            if lhs.startswith("%"):
                # LHS is SSA definition - handle multiple results
                # Cases: "%0 = op", "%0, %1 = op", "%0:2 = op"
                lhs_vars = [x.strip() for x in lhs.split(",")]
                new_lhs_vars = []
                num_results = len(lhs_vars)
                
                for v in lhs_vars:
                    if v.startswith("%"):
                        # Check for :N suffix (e.g., %0:2)
                        if ":" in v:
                            base_var, count_str = v.rsplit(":", 1)
                            if count_str.isdigit():
                                # This is a multi-result binding like %0:2
                                num_results = int(count_str)
                                fresh = builder.fresh("t")
                                ssa_map[base_var] = fresh
                                # Also map the indexed versions %0#0, %0#1, etc.
                                for idx in range(num_results):
                                    ssa_map[f"{base_var}#{idx}"] = f"{fresh}#{idx}"
                                new_lhs_vars.append(f"{fresh}:{num_results}")
                            else:
                                # Not a count suffix, treat as regular var with type annotation
                                fresh = builder.fresh("t")
                                ssa_map[v.split(":")[0]] = fresh
                                new_lhs_vars.append(fresh)
                        else:
                            fresh = builder.fresh("t")
                            ssa_map[v] = fresh
                            new_lhs_vars.append(fresh)
                    else:
                        new_lhs_vars.append(v)
                
                # If we have multiple individual result vars, use :N notation
                if len(lhs_vars) > 1 and not any(":" in v for v in new_lhs_vars):
                    # Multiple comma-separated results: %0, %1 = op
                    # Emit as single binding with :N suffix for cleaner MLIR
                    base_fresh = new_lhs_vars[0]
                    for idx, orig_var in enumerate(lhs_vars):
                        if orig_var.startswith("%"):
                            ssa_map[orig_var] = f"{base_fresh}#{idx}"
                    new_lhs = f"{base_fresh}:{len(lhs_vars)}"
                else:
                    new_lhs = ", ".join(new_lhs_vars)
                
                new_rhs = replace_ssas(rhs, ssa_map)
                new_rhs = replace_affine_maps(new_rhs)
                
                builder.emit(f"{new_lhs} = {new_rhs}")
            else:
                # Not an SSA assignment, emit as-is with SSA replacement
                new_line = replace_ssas(line, ssa_map)
                new_line = replace_affine_maps(new_line)
                builder.emit(new_line)
            continue
            
        # Handle standalone ops (like linalg.yield, cf.assert, etc.)
        new_line = replace_ssas(line, ssa_map)
        new_line = replace_affine_maps(new_line)
        builder.emit(new_line)

    return result_ssa

