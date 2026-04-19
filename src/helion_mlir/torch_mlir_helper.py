"""Compatibility facade for the internal torch-mlir adapter.

New internal code should prefer importing from `helion_mlir.torch_mlir.*`.
This module remains in place to avoid breaking existing callers while the
adapter surface is being migrated behind narrower modules.

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
        
        text = _rewrite_batch_matmul_f16_accumulation(str(module))
        text = _rewrite_amax_keepdim_from_expand_shape(text)
        return text

    def import_node(
        self,
        node: fx.Node,
        input_tensors: list[object],
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
        
        precise_input_values: list[object] = []

        # We need to map args while maintaining structure (lists, tuples)
        def map_graph_arg(arg):
            if isinstance(arg, fx.Node):
                try:
                    val = next(fake_tensor_iter)
                    placeholder = graph.placeholder(f"input_{len(placeholder_nodes)}")
                    fake_val = to_fake_tensor(val)
                    placeholder.meta["val"] = fake_val
                    precise_input_values.append(fake_val)
                    placeholder_nodes.append(placeholder)
                    return placeholder
                except StopIteration:
                    raise RuntimeError("Mismatch between node args and input fake tensors")
            return arg

        new_args = fx.map_arg(node.args, map_graph_arg)
        new_kwargs = fx.map_arg(node.kwargs, map_graph_arg)

        precise_input_iter = iter(precise_input_values)

        def map_value_arg(arg):
            if isinstance(arg, fx.Node):
                try:
                    return next(precise_input_iter)
                except StopIteration:
                    raise RuntimeError("Mismatch between graph args and precise values")
            return arg

        precise_args = fx.map_arg(node.args, map_value_arg)
        precise_kwargs = fx.map_arg(node.kwargs, map_value_arg)
        
        # Create the operation node
        op_node = graph.call_function(node.target, args=new_args, kwargs=new_kwargs)
        
        # Try to infer output type from node metadata
        # Handle both single tensors and tuple outputs (like max.dim)
        import operator
        
        if "val" in node.meta:
            try:
                fake_val = node.target(*precise_args, **precise_kwargs)
            except Exception:
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





def create_fake_tensors_for_node(
    node: fx.Node,
    resolver=None,
) -> list[object]:
    """Create fake tensors for a node's inputs based on metadata.
    
    Args:
        node: FX node whose inputs need fake tensors
        
    Returns:
        List of fake tensor / scalar inputs matching the node's flattened input
        structure.
    """
    from torch._subclasses.fake_tensor import FakeTensorMode
    
    fake_mode = FakeTensorMode()
    fake_tensors = []
    
    def process_arg(arg):
        if isinstance(arg, fx.Node):
            # Try to get existing val
            val = resolver(arg) if resolver is not None else arg.meta.get("val")
            if isinstance(val, torch.Tensor):
                from torch._subclasses.fake_tensor import FakeTensor
                if isinstance(val, FakeTensor):
                    fake_tensors.append(val)
                else:
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
    *,
    input_tensors: list[object] | None = None,
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
    
    fake_tensors = input_tensors if input_tensors is not None else create_fake_tensors_for_node(node)
    return importer.import_node(node, fake_tensors)










import re


def _rewrite_batch_matmul_f16_accumulation(mlir_text: str) -> str:
    """Rewrite linalg.matmul/batch_matmul(f16,f16)->f32+truncf to pure f16 accumulation.

    torch-mlir lowers aten.mm/addmm/bmm/baddbmm with f16 inputs to mixed-precision:
      matmul/batch_matmul with f32 accumulator, then a truncf linalg.generic.

    This rewrites that pattern to use a pure f16 accumulator:
      %cst   = arith.constant 0.000000e+00 : f32  ->  : f16
      %empty = tensor.empty(...) : tensor<...xf32>  ->  : tensor<...xf16>
      %fill  = linalg.fill ins(%cst : f32) ... -> f32  ->  f16
      %mm    = linalg.matmul/batch_matmul ins(f16,f16) outs(f32) -> f32  ->  outs(f16) -> f16
      %trunc = linalg.generic {truncf f32->f16} ...  ->  (removed, %trunc renamed to %mm)
    """
    if 'linalg.batch_matmul' not in mlir_text and 'linalg.matmul' not in mlir_text:
        return mlir_text

    lines = mlir_text.splitlines()

    # Regex to match a matmul or batch_matmul line with f16 inputs and f32 outs
    bmm_re = re.compile(
        r'^(\s*)(%[\w.+-]+)\s*=\s*linalg\.(?:batch_matmul|matmul)\s+'
        r'ins\((?:[^)]*xf16>[^)]*xf16>[^)]*)\)\s+'
        r'outs\((%[\w.+-]+)\s*:\s*(tensor<[^>]+xf32>)\)\s*->\s*(tensor<[^>]+xf32>)\s*$'
    )

    result = list(lines)

    for bmm_idx, line in enumerate(result):
        m = bmm_re.match(line)
        if not m:
            continue

        bmm_result = m.group(2)
        bmm_outs_ssa = m.group(3)
        f32_type = m.group(4)
        f16_type = f32_type.replace('xf32>', 'xf16>')

        # Find the linalg.fill that produced bmm_outs_ssa (search backwards)
        fill_idx = None
        fill_cst_ssa = None
        fill_empty_ssa = None
        fill_re = re.compile(
            r'^\s*' + re.escape(bmm_outs_ssa) + r'\s*=\s*linalg\.fill\s+'
            r'ins\((%[\w.+-]+)\s*:\s*f32\)\s+'
            r'outs\((%[\w.+-]+)\s*:\s*' + re.escape(f32_type) + r'\)\s*->\s*' + re.escape(f32_type) + r'\s*$'
        )
        for j in range(bmm_idx - 1, -1, -1):
            fm = fill_re.match(result[j])
            if fm:
                fill_idx = j
                fill_cst_ssa = fm.group(1)
                fill_empty_ssa = fm.group(2)
                break
        if fill_idx is None:
            continue

        # Find tensor.empty that produced fill_empty_ssa
        empty_idx = None
        empty_re = re.compile(
            r'^\s*' + re.escape(fill_empty_ssa) + r'\s*=\s*tensor\.empty\([^)]*\)\s*:\s*' + re.escape(f32_type) + r'\s*$'
        )
        for j in range(fill_idx - 1, -1, -1):
            if empty_re.match(result[j]):
                empty_idx = j
                break

        # Find arith.constant f32 that produced fill_cst_ssa
        const_idx = None
        const_re = re.compile(
            r'^\s*' + re.escape(fill_cst_ssa) + r'\s*=\s*arith\.constant\s+0\.000000e\+00\s*:\s*f32\s*$'
        )
        for j in range(fill_idx - 1, -1, -1):
            if const_re.match(result[j]):
                const_idx = j
                break

        # Find the truncf linalg.generic that consumes bmm_result (search forwards)
        truncf_start = None
        truncf_end = None
        truncf_result = None
        truncf_re = re.compile(
            r'^\s*(%[\w.+-]+)\s*=\s*linalg\.generic\s*\{[^{}]*\}\s*'
            r'ins\(' + re.escape(bmm_result) + r'\s*:[^)]*xf32>[^)]*\)\s*'
            r'outs\([^)]*xf16>[^)]*\)\s*\{'
        )
        for j in range(bmm_idx + 1, len(result)):
            tm = truncf_re.match(result[j])
            if tm:
                # Verify body contains arith.truncf
                depth = result[j].count('{') - result[j].count('}')
                k = j + 1
                while k < len(result) and depth > 0:
                    depth += result[k].count('{') - result[k].count('}')
                    k += 1
                body_text = '\n'.join(result[j:k])
                if 'arith.truncf' in body_text:
                    truncf_result = tm.group(1)
                    truncf_start = j
                    truncf_end = k - 1
                break

        if truncf_start is None:
            continue

        # Apply rewrites
        # 1. Change arith.constant f32 -> f16
        if const_idx is not None:
            result[const_idx] = result[const_idx].replace(': f32', ': f16', 1)

        # 2. Change tensor.empty f32 -> f16
        if empty_idx is not None:
            result[empty_idx] = result[empty_idx].replace(f32_type, f16_type, 1)

        # 3. Change linalg.fill f32 -> f16 (all f32 in that line are the element type)
        result[fill_idx] = result[fill_idx].replace('f32', 'f16')

        # 4. Change batch_matmul outs f32 -> f16
        result[bmm_idx] = result[bmm_idx].replace(f32_type, f16_type)

        # 5. Remove truncf linalg.generic block
        for k in range(truncf_start, truncf_end + 1):
            result[k] = None

        # 6. Rename all uses of truncf_result to bmm_result
        if truncf_result:
            for k in range(len(result)):
                if result[k] is not None and truncf_result in result[k]:
                    result[k] = re.sub(
                        re.escape(truncf_result) + r'(?![\w.+-])',
                        bmm_result,
                        result[k]
                    )

        # Filter out removed lines
        result = [l for l in result if l is not None]

        # Recurse to handle any additional batch_matmuls
        return _rewrite_batch_matmul_f16_accumulation('\n'.join(result))

    return mlir_text


def _rewrite_amax_keepdim_from_expand_shape(mlir_text: str) -> str:
    """Rewrite torch-mlir's argmax-style amax lowering to a clean keepdim reduction.

    torch-mlir lowers aten.amax(..., keepdim=True) as:
      1. A dual-output linalg.generic producing (max_value: f16, argmax: i64)
         with rank-reduced outs (the reduction axis is dropped).
      2. A tensor.expand_shape that restores the size-1 reduced dim.

    This rewrites that pattern into a single-output linalg.generic whose outs
    indexing map directly keeps the reduced axis as a literal 0
    (e.g. affine_map<(d0, d1, d2) -> (d0, d1, 0)>), matching the form that
    torch-mlir correctly emits for aten.sum(..., keepdim=True).

    Limitations:
      - Only handles single-axis reductions (one "reduction" iterator_type).
      - Only handles maximumf/ogt pattern (amax). amin is not handled.
      - If the argmax result (%R#1) is actually used downstream, the rewrite
        is skipped (e.g. torch.max.dim which returns both values and indices).
    """
    if 'tensor.expand_shape' not in mlir_text:
        return mlir_text
    if ':2' not in mlir_text or 'linalg.generic' not in mlir_text:
        return mlir_text

    lines = mlir_text.splitlines()

    # ----------------------------------------------------------------
    # Pre-pass: collect affine_map alias definitions from the module header.
    # e.g. "#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>"
    # These are needed to resolve alias references inside indexing_maps.
    # ----------------------------------------------------------------
    affine_alias_re = re.compile(r'^(#\w+)\s*=\s*(affine_map<.+>)\s*$')
    alias_to_map: dict[str, str] = {}
    for ln in lines:
        am = affine_alias_re.match(ln.strip())
        if am:
            alias_to_map[am.group(1)] = am.group(2)

    def resolve_map(map_str: str) -> str:
        """Expand a #mapN alias to its inline affine_map<...> form."""
        map_str = map_str.strip()
        return alias_to_map.get(map_str, map_str)

    def extract_dims_from_map(map_str: str):
        """Return (domain_dims, range_exprs) from an affine_map string or #alias."""
        map_str = resolve_map(map_str).strip()
        # Strip "affine_map<...>" wrapper — avoid [^>]+ which stops at "->"
        if map_str.startswith('affine_map<') and map_str.endswith('>'):
            map_str = map_str[len('affine_map<'):-1].strip()
        domain_match = re.match(r'\(([^)]*)\)\s*->', map_str)
        range_match = re.search(r'->\s*\(([^)]*)\)', map_str)
        if not domain_match or not range_match:
            return None, None
        domain_dims = [d.strip() for d in domain_match.group(1).split(',') if d.strip()]
        range_exprs = [e.strip() for e in range_match.group(1).split(',') if e.strip()]
        return domain_dims, range_exprs

    # Regex to match the dual-output linalg.generic header line.
    generic_re = re.compile(
        r'^(\s*)(%[\w.+-]+):2\s*=\s*linalg\.generic\s*\{'
        r'indexing_maps\s*=\s*\[([^\]]+)\]'
        r',\s*iterator_types\s*=\s*\[([^\]]+)\]\}'
        r'\s*ins\(([^)]+)\)'
        r'\s*outs\(([^)]+)\)'
        r'\s*\{\s*$'
    )

    result = list(lines)

    for gen_idx, line in enumerate(result):
        m = generic_re.match(line)
        if not m:
            continue

        indent = m.group(1)
        result_ssa = m.group(2)
        maps_content = m.group(3)
        iter_content = m.group(4)
        ins_content = m.group(5)
        outs_content = m.group(6)

        # Step 1: Exactly one "reduction" iterator_type.
        iter_types = [t.strip().strip('"') for t in iter_content.split(',')]
        if iter_types.count('reduction') != 1:
            continue

        # Step 2: Exactly two outs, second is i64.
        if ':' not in outs_content:
            continue
        outs_ops_part, outs_types_part = outs_content.split(':', 1)
        outs_op_names = [o.strip() for o in outs_ops_part.split(',')]
        outs_type_strs = _split_mlir_types(outs_types_part.strip())
        if len(outs_op_names) != 2 or len(outs_type_strs) != 2:
            continue

        val_fill_ssa = outs_op_names[0]
        idx_fill_ssa = outs_op_names[1]
        val_type_rank_reduced = outs_type_strs[0]
        idx_type_str = outs_type_strs[1]

        if 'xi64' not in idx_type_str:
            continue

        # Step 3: Scan body – validate amax pattern.
        body_start = gen_idx + 1
        depth = 1
        body_end = body_start
        for j in range(body_start, len(result)):
            body_end = j
            depth += result[j].count('{') - result[j].count('}')
            if depth == 0:
                break

        body_text = '\n'.join(result[body_start:body_end + 1])
        required_ops = ['linalg.index', 'arith.index_cast', 'arith.maximumf',
                        'arith.cmpf ogt', 'arith.select']
        if not all(op in body_text for op in required_ops):
            continue
        if 'i64' not in body_text:
            continue

        yield_match = re.search(
            r'linalg\.yield\s+%[\w.+-]+,\s*%[\w.+-]+\s*:\s*(\w+),\s*i64', body_text)
        if not yield_match:
            continue
        elem_type = yield_match.group(1)

        # Step 4: %R#1 must have no downstream users.
        r1_ref = result_ssa + '#1'
        if r1_ref in '\n'.join(result[body_end + 1:]):
            continue

        # Step 5: Locate tensor.expand_shape consuming %R#0.
        r0_ref = result_ssa + '#0'
        expand_idx = None
        expand_result_ssa = None
        keepdim_type = None
        expand_re = re.compile(
            r'^\s*(%[\w.+-]+)\s*=\s*tensor\.expand_shape\s+'
            + re.escape(r0_ref) +
            r'\s+\[([^\]]*(?:\[[^\]]*\][^\]]*)*)\]\s+output_shape\s+\[([^\]]+)\]\s*'
            r':\s*([^\s]+)\s+into\s+([^\s]+)\s*$'
        )
        for j in range(body_end + 1, len(result)):
            em = expand_re.match(result[j])
            if em:
                expand_result_ssa = em.group(1)
                keepdim_type = em.group(5)
                expand_idx = j
                break
            stripped = result[j].strip()
            if stripped.startswith('func.func') or (stripped == '}' and result[j][0] == '}'):
                break

        if expand_idx is None:
            continue

        # Step 6: Derive reduction axis from indexing maps (resolving #aliases).
        all_maps = _split_affine_maps(maps_content)
        if len(all_maps) < 3:
            continue

        ins_domain, ins_range = extract_dims_from_map(all_maps[0])
        _, outs_range = extract_dims_from_map(all_maps[1])

        if ins_domain is None or ins_range is None or outs_range is None:
            continue

        outs_range_set = set(outs_range)
        reduction_dims = [d for d in ins_range if d not in outs_range_set]
        if len(reduction_dims) != 1:
            continue
        reduction_dim_id = reduction_dims[0]

        if reduction_dim_id not in ins_domain:
            continue
        reduction_axis = ins_domain.index(reduction_dim_id)

        # Step 7: Build keepdim outs indexing map.
        new_outs_range = list(outs_range)
        new_outs_range.insert(reduction_axis, '0')
        new_outs_map = (
            f'affine_map<({", ".join(ins_domain)}) -> ({", ".join(new_outs_range)})>'
        )

        # Step 8: Build ins map string (resolve alias to inline form for the new header).
        ins_map_inline = resolve_map(all_maps[0].strip())
        new_indexing_maps = f'[{ins_map_inline}, {new_outs_map}]'

        # Step 9: Locate i64 producers (tensor.empty + linalg.fill).
        idx_empty_ssa = None
        idx_fill_idx = None
        idx_empty_idx = None

        fill_re_i64 = re.compile(
            r'^\s*' + re.escape(idx_fill_ssa) + r'\s*=\s*linalg\.fill\s+'
            r'ins\((%[\w.+-]+)\s*:\s*i64\)\s+'
            r'outs\((%[\w.+-]+)\s*:\s*' + re.escape(idx_type_str) + r'\)\s*->\s*'
            + re.escape(idx_type_str) + r'\s*$'
        )
        for j in range(gen_idx - 1, -1, -1):
            fm = fill_re_i64.match(result[j])
            if fm:
                idx_fill_idx = j
                idx_empty_ssa = fm.group(2)
                break

        if idx_fill_idx is not None and idx_empty_ssa is not None:
            empty_re_i64 = re.compile(
                r'^\s*' + re.escape(idx_empty_ssa)
                + r'\s*=\s*tensor\.empty\([^)]*\)\s*:\s*' + re.escape(idx_type_str) + r'\s*$'
            )
            for j in range(idx_fill_idx - 1, -1, -1):
                if empty_re_i64.match(result[j]):
                    idx_empty_idx = j
                    break

        full_text_check = '\n'.join(r for r in result if r is not None)
        can_drop_i64_fill = (
            idx_fill_idx is not None
            and full_text_check.count(idx_fill_ssa) == 2
        )
        can_drop_i64_empty = (
            idx_empty_idx is not None and idx_empty_ssa is not None
            and full_text_check.count(idx_empty_ssa) == 2
        )

        # Step 10: Locate value-side empty/fill producers.
        val_fill_idx = None
        val_empty_ssa = None
        val_empty_idx = None

        fill_re_val = re.compile(
            r'^\s*' + re.escape(val_fill_ssa) + r'\s*=\s*linalg\.fill\s+'
            r'ins\((%[\w.+-]+)\s*:\s*' + re.escape(elem_type) + r'\)\s+'
            r'outs\((%[\w.+-]+)\s*:\s*' + re.escape(val_type_rank_reduced) + r'\)\s*->\s*'
            + re.escape(val_type_rank_reduced) + r'\s*$'
        )
        for j in range(gen_idx - 1, -1, -1):
            fm = fill_re_val.match(result[j])
            if fm:
                val_fill_idx = j
                val_empty_ssa = fm.group(2)
                break

        if val_fill_idx is not None and val_empty_ssa is not None:
            empty_re_val = re.compile(
                r'^\s*' + re.escape(val_empty_ssa)
                + r'\s*=\s*tensor\.empty\([^)]*\)\s*:\s*' + re.escape(val_type_rank_reduced) + r'\s*$'
            )
            for j in range(val_fill_idx - 1, -1, -1):
                if empty_re_val.match(result[j]):
                    val_empty_idx = j
                    break

        # Step 11: Apply all mutations.

        # 11a. Retype value-side tensor.empty and linalg.fill.
        if val_empty_idx is not None:
            result[val_empty_idx] = result[val_empty_idx].replace(
                val_type_rank_reduced, keepdim_type, 1)
        if val_fill_idx is not None:
            result[val_fill_idx] = result[val_fill_idx].replace(
                val_type_rank_reduced, keepdim_type)

        # 11b. Rewrite the linalg.generic header (single output, inline maps).
        result[gen_idx] = (
            f'{indent}{result_ssa} = linalg.generic {{'
            f'indexing_maps = {new_indexing_maps}, '
            f'iterator_types = [{iter_content}]}} '
            f'ins({ins_content}) '
            f'outs({val_fill_ssa} : {keepdim_type}) {{'
        )

        # 11c. Rewrite body: strip i64 tracking, simplify block args and yield.
        for j in range(body_start, body_end + 1):
            stripped = result[j].strip()
            if stripped.startswith('^bb0'):
                bb_m = re.match(r'(\^bb0\()(.*)(\):)', stripped)
                if bb_m:
                    new_args = [a.strip() for a in bb_m.group(2).split(',')
                                if 'i64' not in a]
                    result[j] = (result[j][:result[j].index('^')]
                                 + f'^bb0({", ".join(new_args)}):')
            elif 'linalg.index' in stripped:
                result[j] = None
            elif 'arith.index_cast' in stripped and 'index to i64' in stripped:
                result[j] = None
            elif 'arith.cmpf ogt' in stripped:
                result[j] = None
            elif 'arith.select' in stripped and 'i64' in stripped:
                result[j] = None
            elif stripped.startswith('linalg.yield'):
                ym = re.search(
                    r'linalg\.yield\s+(%[\w.+-]+),\s*%[\w.+-]+\s*:\s*\w+,\s*i64', stripped)
                if ym:
                    result[j] = (result[j][:result[j].index('linalg.yield')]
                                 + f'linalg.yield {ym.group(1)} : {elem_type}')
            elif stripped.startswith('}') and '->' in result[j]:
                result[j] = result[j][:result[j].index('}')+1] + f' -> {keepdim_type}'

        # 11d. Drop i64 producers if safe.
        if can_drop_i64_fill:
            result[idx_fill_idx] = None
        if can_drop_i64_empty:
            result[idx_empty_idx] = None

        # 11e. Drop expand_shape.
        result[expand_idx] = None

        # 11f. Rename expand_shape result -> result_ssa; normalise %R#0 -> %R.
        ssa_rename = re.compile(re.escape(expand_result_ssa) + r'(?![\w.+-])')
        r0_rename = re.compile(re.escape(r0_ref) + r'(?![\w.+-])')
        for j in range(len(result)):
            if result[j] is None:
                continue
            if expand_result_ssa in result[j]:
                result[j] = ssa_rename.sub(result_ssa, result[j])
            if r0_ref in result[j]:
                result[j] = r0_rename.sub(result_ssa, result[j])

        result = [l for l in result if l is not None]
        return _rewrite_amax_keepdim_from_expand_shape('\n'.join(result))

    return mlir_text


def _rewrite_linalg_generic_scalars(
    rhs: str,
    scalar_arg_ssas: dict[str, str],
    ssa_map: dict[str, str],
    pending_scalar_bb_indices: list,
) -> str:
    """Rewrite a linalg.generic RHS to remove scalar operands from ins().

    Scalar operands (identified by scalar_arg_ssas) are removed from the
    ins() clause and their corresponding indexing_maps entries. The positions
    of removed block args are recorded in pending_scalar_bb_indices so the
    caller can skip them when processing the next ^bb0 line.

    Args:
        rhs: The RHS of the assignment (everything after '=').
        scalar_arg_ssas: Maps original %argN names to scalar SSA values.
        ssa_map: Current SSA name mapping (for resolving %argN -> operand SSA).
        pending_scalar_bb_indices: Output list of (bb_position, scalar_ssa) tuples.

    Returns:
        Rewritten RHS with scalar operands removed from ins() and indexing_maps.
    """
    # Find ins(...) clause and parse its operands
    ins_match = re.search(r'ins\(([^)]*)\)', rhs)
    if not ins_match:
        return rhs

    ins_content = ins_match.group(1)
    # ins content looks like: %arg0, %arg1 : tensor<?x?xf16>, tensor<f16>
    if ':' not in ins_content:
        return rhs

    operands_part, types_part = ins_content.split(':', 1)
    # Parse operand names (before the colon)
    operand_names = [o.strip() for o in operands_part.split(',')]

    # Parse type list (after the colon) - need to handle nested <> in types
    type_strs = _split_mlir_types(types_part.strip())

    if len(operand_names) != len(type_strs):
        return rhs  # Can't parse, leave unchanged

    # Identify which positions are scalar
    scalar_positions = set()
    for pos, op_name in enumerate(operand_names):
        if op_name in scalar_arg_ssas:
            scalar_positions.add(pos)
            # The scalar is at position `pos` in ins(), which is also position
            # `pos` in ^bb0 block args (outs args come after ins args).
            scalar_ssa = scalar_arg_ssas[op_name]
            pending_scalar_bb_indices.append((pos, scalar_ssa))

    if not scalar_positions:
        return rhs  # No scalars found

    # Rebuild ins() without scalar entries
    new_operands = []
    new_types = []
    for pos in range(len(operand_names)):
        if pos not in scalar_positions:
            new_operands.append(operand_names[pos])
            new_types.append(type_strs[pos])

    new_ins = f"ins({', '.join(new_operands)} : {', '.join(new_types)})"

    # Remove corresponding indexing_maps entries
    # indexing_maps has entries for: [ins_0, ins_1, ..., outs_0, outs_1, ...]
    # We need to remove entries at the scalar positions
    maps_match = re.search(r'indexing_maps\s*=\s*\[([^\]]*)\]', rhs)
    if maps_match:
        maps_content = maps_match.group(1)
        # Split affine maps carefully (they contain commas inside <>)
        map_entries = _split_affine_maps(maps_content)

        new_map_entries = []
        for pos, entry in enumerate(map_entries):
            if pos not in scalar_positions:
                new_map_entries.append(entry)

        new_maps = f"indexing_maps = [{', '.join(new_map_entries)}]"
        rhs = rhs[:maps_match.start()] + new_maps + rhs[maps_match.end():]

    # Replace ins(...) in the (possibly already modified) rhs
    ins_match = re.search(r'ins\([^)]*\)', rhs)
    if ins_match:
        rhs = rhs[:ins_match.start()] + new_ins + rhs[ins_match.end():]

    return rhs


def _split_mlir_types(types_str: str) -> list[str]:
    """Split a comma-separated list of MLIR types, respecting nested <>.

    Example: "tensor<?x?xf16>, tensor<f16>" -> ["tensor<?x?xf16>", "tensor<f16>"]
    """
    result = []
    depth = 0
    current = []
    for ch in types_str:
        if ch == '<':
            depth += 1
            current.append(ch)
        elif ch == '>':
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            result.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        result.append(''.join(current).strip())
    return result


def _split_affine_maps(maps_str: str) -> list[str]:
    """Split a comma-separated list of affine_map entries, respecting nested <>.

    Example: "affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>"
    -> ["affine_map<(d0) -> (d0)>", "affine_map<(d0) -> ()>"]
    """
    return _split_mlir_types(maps_str)  # Same parsing logic


def inline_torch_mlir_output(
    mlir_text: str,
    operands: list[str],
    mlir_output_helper,
    *,
    dimension_ssa_map: dict[str, list[str | None]] | None = None,
    scalar_operand_map: dict[int, str] | None = None,
) -> str:
    """Inline torch-mlir generated text into the current output helper.

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
        mlir_output_helper: The MLIR helper to emit to.
        dimension_ssa_map: Optional mapping from operand SSA to list of dimension SSAs.
            If provided, tensor.dim operations are replaced with pre-existing SSAs.
            Each list entry is either:
            - str: Use this SSA value for the dimension
            - None: Dimension is static, cannot be queried (or let tensor.dim run)
            Example: {"%slice16": ["%block_size_0", "%block_size_1", None]}
        scalar_operand_map: Optional mapping from operand index to scalar SSA value.
            Scalar operands are removed from linalg.generic ins() and their
            corresponding block args are replaced with direct scalar SSA references
            in the body. This matches the pattern where scalar constants are
            captured directly in linalg.generic body rather than passed as
            0-rank tensor ins() operands.


    Returns:
        The SSA value of the result.
    """
    lines = mlir_text.splitlines()
    ssa_map = {}  # Maps original SSA values to renamed ones
    
    # -------------------------------------------------------------------------
    # PASS 1: Collect and rename affine map aliases
    # -------------------------------------------------------------------------
    # Torch-mlir emits affine_map definitions like: #map = affine_map<...>
    # We inline these definitions directly where they're used to avoid 
    # namespace collisions when multiple ATen ops are inlined.
    
    # Track affine map aliases to inline them
    affine_map_aliases = {}  # old_alias -> (new_alias, affine_map_def)
    affine_map_counter = getattr(mlir_output_helper, '_affine_map_counter', 0)
    
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
    
    mlir_output_helper._affine_map_counter = affine_map_counter
    
    # Helper to replace affine map aliases with inline definitions
    # Use regex to ensure we match whole aliases (e.g., #map but not #map1)
    def replace_affine_maps(text):
        for old_alias, (new_alias, affine_map_def) in affine_map_aliases.items():
            # Escape the alias for regex and use word boundary to prevent partial matches
            # e.g., #map should not match inside #map1
            pattern = re.escape(old_alias) + r'(?![0-9a-zA-Z_])'
            text = re.sub(pattern, affine_map_def, text)
        return text
    
    # -------------------------------------------------------------------------
    # PASS 2: Map function arguments to caller's SSA values
    # -------------------------------------------------------------------------
    for i, op in enumerate(operands):
        ssa_map[f"%arg{i}"] = op

    # -------------------------------------------------------------------------
    # Track scalar operands for linalg.generic body capture
    # -------------------------------------------------------------------------
    scalar_arg_ssas = {}
    if scalar_operand_map:
        for idx, scalar_ssa in scalar_operand_map.items():
            scalar_arg_ssas[f"%arg{idx}"] = scalar_ssa

    # Regex to identify SSAs
    ssa_pattern = re.compile(r'%([a-zA-Z0-9_][a-zA-Z0-9_.+-]*)')

    def replace_ssas(text, mapping):
        def repl(m):
            name = m.group(1)
            full = f"%{name}"
            return mapping.get(full, full)
        return ssa_pattern.sub(repl, text)

    # -------------------------------------------------------------------------
    # OPTIMIZATION SUPPORT: Track arith.constant index values for tensor.dim
    # -------------------------------------------------------------------------
    # When dimension_ssa_map is provided, tensor.dim can be replaced by
    # pre-existing dimension SSAs. We still track integer index constants to
    # decode the tensor.dim index, but we must emit constants eagerly because
    # they may also be used by non-tensor.dim ops (e.g. repeat lowering).
    #
    # Track arith.constant values for tensor.dim index resolution
    const_index_values = {}  # SSA name (e.g., "%0") -> integer value
    
    # Deferred arith.constant emissions - only emit if not consumed by tensor.dim
    deferred_const_emissions = {}  # original SSA name -> (fresh_name, rhs_text)

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

    # Track which ^bb0 block arg positions to skip for scalar operands.
    # Populated when processing a linalg.generic ins() line, consumed
    # when processing the next ^bb0 line.
    pending_scalar_bb_indices = []  # list of (position_in_bb0, scalar_ssa)

    # -------------------------------------------------------------------------
    # PASS 3: Process function body lines and rename SSA values
    # -------------------------------------------------------------------------
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
            mlir_output_helper.emit(new_line)
            continue
            
        # Handle Block Args (e.g. ^bb0(%a: f32, %b: f32):)
        if line.startswith("^"):
            if "(" in line and ")" in line:
                pre, rest = line.split("(", 1)
                args_part, post = rest.rsplit(")", 1)

                args_list = args_part.split(",")
                # Build set of positions to skip (scalar operands)
                skip_positions = {}
                for pos, scalar_ssa in pending_scalar_bb_indices:
                    skip_positions[pos] = scalar_ssa
                pending_scalar_bb_indices.clear()

                new_args_list = []
                for arg_idx, arg_def in enumerate(args_list):
                    arg_def = arg_def.strip()
                    if arg_idx in skip_positions:
                        # This block arg corresponds to a scalar operand.
                        # Map it to the scalar SSA instead of creating a block arg.
                        if ":" in arg_def:
                            name_part, _ = arg_def.split(":", 1)
                            name = name_part.strip()
                            if name.startswith("%"):
                                ssa_map[name] = skip_positions[arg_idx]
                        continue  # Skip this block arg
                    if ":" in arg_def:
                        name_part, type_part = arg_def.split(":", 1)
                        name = name_part.strip()
                        if name.startswith("%"):
                            fresh = mlir_output_helper.fresh("blk_arg")
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
            mlir_output_helper.emit(new_line)
            continue

        # =====================================================================
        # Handle Assignment lines (SSA definitions with '=')
        # =====================================================================
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
                
                # -----------------------------------------------------------------
                # Track arith.constant index values for tensor.dim.
                # Pattern: %0 = arith.constant 0 : index
                #
                # IMPORTANT:
                # Do not defer emission of index constants. They can be used by
                # non-tensor.dim operations (seen in repeat lowering), and
                # deferring can produce undefined SSA references.
                # -----------------------------------------------------------------
                arith_const_match = re.match(r'arith\.constant\s+(\d+)\s*:\s*index', rhs)
                if arith_const_match and len(lhs_vars) == 1:
                    const_val = int(arith_const_match.group(1))
                    ssa_name = lhs_vars[0]
                    const_index_values[ssa_name] = const_val
                    # Create a fresh name mapping and emit eagerly.
                    fresh = mlir_output_helper.fresh("t")
                    ssa_map[ssa_name] = fresh
                    mlir_output_helper.emit(f"{fresh} = {rhs}")
                    continue
                
                # -----------------------------------------------------------------
                # Optimization: Replace tensor.dim with pre-existing dimension SSA
                # Pattern: %N = tensor.dim %arg0, %const : tensor<...>
                # -----------------------------------------------------------------
                tensor_dim_match = re.match(r'tensor\.dim\s+(%\w+),\s*(%\w+)\s*:', rhs)
                if tensor_dim_match and dimension_ssa_map:
                    tensor_ssa_ref = tensor_dim_match.group(1)
                    const_ssa_ref = tensor_dim_match.group(2)
                    
                    # Resolve the tensor SSA to our operand name
                    tensor_ssa_resolved = ssa_map.get(tensor_ssa_ref, tensor_ssa_ref)
                    
                    # Get the dimension index from the constant
                    dim_idx = const_index_values.get(const_ssa_ref)
                    
                    # Check if we have a pre-existing SSA for this dimension
                    if tensor_ssa_resolved in dimension_ssa_map and dim_idx is not None:
                        dim_ssas = dimension_ssa_map[tensor_ssa_resolved]
                        if dim_idx < len(dim_ssas) and dim_ssas[dim_idx] is not None:
                            # We have a pre-existing SSA - use it!
                            result_ssa_name = lhs_vars[0]
                            ssa_map[result_ssa_name] = dim_ssas[dim_idx]
                            # Mark the index constant as consumed (don't emit it)
                            if const_ssa_ref in deferred_const_emissions:
                                del deferred_const_emissions[const_ssa_ref]
                            # Skip emitting this tensor.dim operation
                            continue
                    
                    # Fallback: emit the tensor.dim normally
                    # First, emit the deferred constant if it exists
                    if const_ssa_ref in deferred_const_emissions:
                        fresh_name, const_rhs = deferred_const_emissions.pop(const_ssa_ref)
                        mlir_output_helper.emit(f"{fresh_name} = {const_rhs}")
                
                # ---------------------------------------------------------
                # Normal SSA assignment processing: generate fresh names 
                # and handle multi-result bindings (e.g., %0:2 = op)
                # ---------------------------------------------------------
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
                                fresh = mlir_output_helper.fresh("t")
                                ssa_map[base_var] = fresh
                                # Also map the indexed versions %0#0, %0#1, etc.
                                for idx in range(num_results):
                                    ssa_map[f"{base_var}#{idx}"] = f"{fresh}#{idx}"
                                new_lhs_vars.append(f"{fresh}:{num_results}")
                            else:
                                # Not a count suffix, treat as regular var with type annotation
                                fresh = mlir_output_helper.fresh("t")
                                ssa_map[v.split(":")[0]] = fresh
                                new_lhs_vars.append(fresh)
                        else:
                            fresh = mlir_output_helper.fresh("t")
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
                
                # ---------------------------------------------------------
                # Scalar operand rewriting for linalg.generic
                # Remove scalar operands from ins() and indexing_maps,
                # and track their ^bb0 block arg positions for removal.
                # ---------------------------------------------------------
                if scalar_arg_ssas and 'linalg.generic' in rhs and 'ins(' in rhs:
                    rhs = _rewrite_linalg_generic_scalars(
                        rhs, scalar_arg_ssas, ssa_map, pending_scalar_bb_indices
                    )

                new_rhs = replace_ssas(rhs, ssa_map)
                new_rhs = replace_affine_maps(new_rhs)

                mlir_output_helper.emit(f"{new_lhs} = {new_rhs}")
            else:
                # Not an SSA assignment, emit as-is with SSA replacement
                new_line = replace_ssas(line, ssa_map)
                new_line = replace_affine_maps(new_line)
                mlir_output_helper.emit(new_line)
            continue
            
        # Handle standalone ops (like linalg.yield, cf.assert, etc.)
        new_line = replace_ssas(line, ssa_map)
        new_line = replace_affine_maps(new_line)
        mlir_output_helper.emit(new_line)

    return result_ssa
