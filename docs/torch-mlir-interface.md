# Torch-MLIR Interface

This document explains how `helion_mlir` uses torch-mlir to convert ATen operations to MLIR, including pre-processing of Helion's FX nodes and post-processing of torch-mlir's output.

## Overview

The `torch_mlir_helper.py` module bridges Helion's Device IR with torch-mlir's `FxImporter` infrastructure. It handles:

1. **Pre-processing**: Normalizing Helion FX nodes to match torch-mlir's expected format
2. **MLIR Generation**: Using `FxImporter` to convert ATen ops to MLIR
3. **Post-processing**: Extracting operations from torch-mlir's output and inlining them with proper SSA renaming

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ATen Operation Flow                                   │
│                                                                             │
│  Helion Device IR Node (FX)                                                 │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  1. PRE-PROCESSING: normalize_helion_node()                         │   │
│  │     - Synthesize tensor_meta from node.meta['val']                  │   │
│  │     - Convert concrete tensors to meta tensors                      │   │
│  │     - Handle tuple outputs (e.g., from max.dim)                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  2. MLIR GENERATION: TorchMLIRNodeImporter.import_node()            │   │
│  │     - Create minimal FX graph with placeholders                     │   │
│  │     - Use torch-mlir's FxImporter to generate MLIR                  │   │
│  │     - Optionally lower via pipelines (linalg-on-tensors, tosa, etc.)│   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  3. POST-PROCESSING: inline_torch_mlir_output()                     │   │
│  │     - Parse torch-mlir's MLIR module text                           │   │
│  │     - Extract function body ops                                     │   │
│  │     - Rename SSA values (%arg0 → operand SSAs)                      │   │
│  │     - Inline affine map definitions                                 │   │
│  │     - Emit into current MLIRBuilder                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│           │                                                                 │
│           ▼                                                                 │
│       Result SSA                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Pre-Processing: Helion Node Normalization

### The Problem

Torch-mlir's `FxImporter` expects FX nodes to have specific metadata fields that standard PyTorch FX produces:
- `node.meta['tensor_meta']`: Shape, dtype, stride, memory_format, etc.
- `node.meta['val']`: FakeTensor (not concrete tensors)

Helion's Device IR uses `node.meta['val']` with potentially concrete tensors or different metadata formats.

### Solution: `normalize_helion_node()`

This function ensures Helion nodes are compatible with torch-mlir:

```python
def normalize_helion_node(node: fx.Node) -> fx.Node:
    """Normalize a Helion Device IR node for torch-mlir compatibility."""
```

**Key transformations:**

| Helion Format | Torch-MLIR Expected | Transformation |
|--------------|---------------------|----------------|
| `meta['val']` = concrete tensor | `meta['val']` = FakeTensor | Convert via `torch.empty(..., device='meta')` |
| No `tensor_meta` | `tensor_meta` dict | Synthesize from `val` using `synthesize_tensor_meta()` |
| Tuple output (e.g., `max.dim`) | Tuple of FakeTensors | Convert each element, create tuple of `tensor_meta` |

### `synthesize_tensor_meta()`

Creates the required metadata dictionary from a tensor:

```python
def synthesize_tensor_meta(val: torch.Tensor) -> dict:
    return {
        'shape': tuple(val.shape),
        'dtype': val.dtype,
        'requires_grad': val.requires_grad,
        'stride': tuple(val.stride()),
        'memory_format': torch.contiguous_format,  # or channels_last
        'is_quantized': False,
    }
```

---

## MLIR Generation: TorchMLIRNodeImporter

### Class Overview

```python
class TorchMLIRNodeImporter:
    """Imports FX nodes to MLIR using torch-mlir's FxImporter."""
    
    def __init__(self, output_type: str = "raw"):
        """
        Args:
            output_type: Target MLIR dialect
                - "raw": torch dialect (torch.aten.*)
                - "linalg-on-tensors": linalg dialect
                - "tosa": TOSA dialect
                - "stablehlo": StableHLO dialect
        """
```

### Key Methods

#### `import_node(node, input_tensors)`

Imports a single FX node by wrapping it in a minimal graph:

1. **Create placeholder nodes** for each input tensor
2. **Call the target operation** with mapped args
3. **Handle tuple outputs** by adding `getitem` nodes
4. **Import the graph** via `FxImporter.import_stateless_graph()`

```python
def import_node(self, node: fx.Node, input_tensors: list[torch.Tensor]) -> str:
    # Create a minimal FX graph containing just this node
    graph = fx.Graph()
    
    # Create placeholder nodes for inputs
    for val in input_tensors:
        placeholder = graph.placeholder(f"input_{i}")
        placeholder.meta["val"] = to_fake_tensor(val)
    
    # Create the operation node
    op_node = graph.call_function(node.target, args=new_args, kwargs=new_kwargs)
    
    # Handle tuple outputs (e.g., max.dim returns (values, indices))
    if isinstance(output_val, tuple):
        # Decompose tuple output using getitem nodes
        for i, elem_val in enumerate(output_val):
            getitem_node = graph.call_function(operator.getitem, (op_node, i))
            getitem_nodes.append(getitem_node)
        graph.output(tuple(getitem_nodes))
    else:
        graph.output(op_node)
    
    return self.import_graph(graph)
```

### Lowering Pipeline

For non-"raw" output types, torch-mlir applies two-stage lowering:

```python
# Stage 1: RAW → Torch Backend IR
run_pipeline_with_repro_report(
    module,
    "builtin.module(func.func(torch-match-quantized-custom-ops), "
    "torchdynamo-export-to-torch-backend-pipeline{ extra-library=})",
    "Lowering TorchFX IR -> Torch Backend IR"
)

# Stage 2: Torch Backend IR → Target Dialect
module = lower_mlir_module(
    verbose=False,
    output_type=OutputType.get("linalg-on-tensors"),  # or tosa, stablehlo
    module=module
)
```

---

## Post-Processing: MLIR Inlining

### The Problem

Torch-mlir produces a complete MLIR module with its own function:

```mlir
module {
  #map = affine_map<(d0, d1) -> (d0, d1)>
  func.func @aten_op(%arg0: tensor<32x64xf32>, %arg1: tensor<32x64xf32>) -> tensor<32x64xf32> {
    %0 = linalg.generic {...} ins(%arg0, %arg1 : ...) outs(...) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %1 = arith.addf %in, %in_0 : f32
        linalg.yield %1 : f32
    } -> tensor<32x64xf32>
    return %0 : tensor<32x64xf32>
  }
}
```

We need to **extract** the operations and **inline** them into our existing MLIR with:
- Proper SSA renaming (`%arg0` → actual operand SSAs)
- Affine map inlining (no module-level aliases)
- Fresh SSA names for all defined values

### Solution: `inline_torch_mlir_output()`

```python
def inline_torch_mlir_output(
    mlir_text: str, 
    operands: list[str],  # Actual SSA values for function args
    builder: MLIRBuilder
) -> str:
    """Extract ops from torch-mlir module and inline into current builder."""
```

**Processing steps:**

1. **Collect affine map definitions**: Parse `#map = affine_map<...>` lines
2. **Map function arguments**: `%arg0` → `operands[0]`, etc.
3. **Process each line in function body**:
   - Skip affine map definitions (already inlined)
   - Handle block args (`^bb0(%a: f32, ...)`) with fresh SSA names
   - Handle assignments (`%x = op ...`) with fresh SSA and operand replacement
   - Handle standalone ops (`linalg.yield`, closing braces)
4. **Return the result SSA** from the return statement

### SSA Renaming Example

**Input from torch-mlir:**
```mlir
%0 = linalg.generic {...} ins(%arg0, %arg1 : ...) outs(...) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.addf %in, %in_0 : f32
    linalg.yield %1 : f32
} -> tensor<32x64xf32>
```

**After inlining (with operands = [`%tensor_a`, `%tensor_b`]):**
```mlir
%t0 = linalg.generic {...} ins(%tensor_a, %tensor_b : ...) outs(...) {
  ^bb0(%blk_arg0: f32, %blk_arg1: f32, %blk_arg2: f32):
    %t1 = arith.addf %blk_arg0, %blk_arg1 : f32
    linalg.yield %t1 : f32
} -> tensor<32x64xf32>
```

### Affine Map Inlining

Torch-mlir defines affine maps at module level (`#map = affine_map<...>`). Since we're inlining into an existing module, we must inline the definitions:

**Before:**
```mlir
#map = affine_map<(d0, d1) -> (d0, d1)>
%0 = linalg.generic {indexing_maps = [#map, #map], ...}
```

**After inlining:**
```mlir
%t0 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, 
                                        affine_map<(d0, d1) -> (d0, d1)>], ...}
```

---

## Integration with IRVisitor

The `visit_aten_compute()` method in `ir_visitor.py` ties everything together:

```python
def visit_aten_compute(self, node: fx.Node) -> str:
    """Generate MLIR for ATen compute ops using torch-mlir."""
    
    # Get output type (defaults to "raw" for torch dialect)
    output_type = getattr(self.ctx, 'aten_output_type', 'raw')
    
    # Use torch-mlir to generate MLIR for this operation
    mlir_text = import_aten_node_to_mlir(node, output_type=output_type)
    
    # Collect SSA values for tensor operands
    tensor_operands = []
    def collect_operands(arg):
        if isinstance(arg, fx.Node):
            tensor_operands.append(self.ctx.node_values.get(arg.name))
        return arg
    fx.map_arg(node.args, collect_operands)
    fx.map_arg(node.kwargs, collect_operands)
    
    # Inline the generated MLIR
    result = inline_torch_mlir_output(mlir_text, tensor_operands, self.builder)
    
    self.ctx.node_values[node.name] = result
    return result
```

---

## Supported Output Types

| Output Type | MLIR Dialect | Use Case |
|-------------|--------------|----------|
| `raw` | `torch.aten.*` | Debugging, torch-specific passes |
| `linalg-on-tensors` | `linalg.generic`, `linalg.matmul`, etc. | Hardware-agnostic lowering |
| `tosa` | `tosa.*` | ARM/mobile targets |
| `stablehlo` | `stablehlo.*` | XLA/TPU compatibility |

---

## Example: `aten.addmm` Flow

**Input FX Node:**
```python
Node(target=aten.addmm.default, args=(bias, input, weight))
```

**Step 1: Pre-processing**
- Extract tensor shapes from `node.meta['val']`
- Create FakeTensors for inputs

**Step 2: MLIR Generation (linalg-on-tensors)**
```mlir
module {
  func.func @aten_op(%arg0: tensor<64xf32>, %arg1: tensor<32x64xf32>, 
                      %arg2: tensor<64x128xf32>) -> tensor<32x128xf32> {
    %0 = linalg.matmul ins(%arg1, %arg2 : ...) outs(...) -> tensor<32x128xf32>
    %1 = linalg.generic {...} ins(%0, %arg0 : ...) outs(...) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %2 = arith.addf %in, %in_0 : f32
        linalg.yield %2 : f32
    } -> tensor<32x128xf32>
    return %1 : tensor<32x128xf32>
  }
}
```

**Step 3: Post-processing**
- `%arg0` → `%bias_ssa`
- `%arg1` → `%input_ssa`
- `%arg2` → `%weight_ssa`
- All `%N` values renamed to fresh `%tN`

**Final Inlined MLIR:**
```mlir
%t0 = linalg.matmul ins(%input_ssa, %weight_ssa : ...) outs(...) -> tensor<32x128xf32>
%t1 = linalg.generic {...} ins(%t0, %bias_ssa : ...) outs(...) {
  ^bb0(%blk_arg0: f32, %blk_arg1: f32, %blk_arg2: f32):
    %t2 = arith.addf %blk_arg0, %blk_arg1 : f32
    linalg.yield %t2 : f32
} -> tensor<32x128xf32>
```

---

## Key Files

| File | Purpose |
|------|---------|
| [torch_mlir_helper.py](file:///home/zhenyu/helion-mlir/src/helion_mlir/torch_mlir_helper.py) | Main interface between Helion and torch-mlir |
| [ir_visitor.py](file:///home/zhenyu/helion-mlir/src/helion_mlir/ir_visitor.py) | Uses `import_aten_node_to_mlir()` and `inline_torch_mlir_output()` |
| [compiler_utils.py](file:///home/zhenyu/helion-mlir/.venv/lib/python3.11/site-packages/torch_mlir/compiler_utils.py) | Torch-mlir's `OutputType`, `run_pipeline_with_repro_report()`, `lower_mlir_module()` |
| [fx_importer.py](file:///home/zhenyu/helion-mlir/.venv/lib/python3.11/site-packages/torch_mlir/extras/fx_importer.py) | Torch-mlir's `FxImporter` class |
