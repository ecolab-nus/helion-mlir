# Helion FX-to-MLIR Software Architecture

This document describes the software architecture of the `helion_fx_mlir` package, which converts Helion Device IR (FX graphs) to MLIR text representation.

## Overview

The `helion_fx_mlir` package uses a **visitor pattern** to walk Device IR FX graphs node-by-node. It generates:
1. **Standard MLIR dialects** (`tensor`, `affine`, `linalg`, `arith`) for Helion-specific operations
2. **Torch dialect** (`torch.aten.*`) via **torch-mlir** for ATen operations

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        BoundKernel (Helion)                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │
│  │ fake_args   │  │    env      │  │        host_function        │  │
│  │ (Tensors)   │  │(BlockSizes) │  │  └─> device_ir (DeviceIR)   │  │
│  └─────────────┘  └─────────────┘  │      └─> graphs [GraphInfo] │  │
│                                    │          └─> graph (FX)     │  │
│                                    └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         generate_mlir()                             │
│                        (helion_mlir.py)                             │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌───────────┐   ┌──────────────┐   ┌──────────────┐
            │ IRVisitor │   │LoweringContext│  │  MLIRBuilder │
            │  (Walk)   │   │   (State)     │  │  (Emission)  │
            └───────────┘   └──────────────┘   └──────────────┘
                    │                               │
        ┌───────────┴────────────────────┬──────────┘
        │                                │
        ▼                                ▼
┌─────────────────────┐      ┌───────────────────────┐
│   Helion Ops        │      │   ATen Ops            │
│   (visit_* methods) │      │   (torch_mlir_helper) │
└─────────────────────┘      └───────────────────────┘
                    │                   │
                    └───────┬───────────┘
                            ▼
                    ┌───────────────┐
                    │  MLIR Text    │
                    └───────────────┘
```

---

## Package Structure

```
src/helion_fx_mlir/
├── __init__.py              # Public API exports
├── helion_mlir.py           # Main entry: generate_mlir(), validate_with_helion_opt()
├── ir_visitor.py            # IRVisitor: walks FX graphs, dispatches to visit_* methods
├── lowering_context.py      # LoweringContext: state (loops, args, SSA mappings)
├── mlir_builder.py          # MLIRBuilder: text emission, SSA naming, indentation
└── torch_mlir_helper.py     # torch-mlir integration for ATen ops
```

---

## Core Modules

### 1. `helion_mlir.py` — Entry Point

**Purpose**: Main entry point for MLIR generation.

**Key Functions**:
- `generate_mlir(bound_kernel, kernel_name)` → MLIR text
- `validate_with_helion_opt(mlir_text)` → validates via `helion-opt` or `mlir-opt`
- `_collect_host_tensor_names()` → pre-scans graphs for tensor function parameters

### 2. `ir_visitor.py` — Graph Walker

**Purpose**: Walks Device IR FX graphs and generates MLIR via `visit_*` methods.

**Key Methods**:
| Method | Device IR Target | Generated MLIR |
|--------|------------------|----------------|
| `visit_load` | `helion.language.memory_ops.load` | `tensor.extract_slice` |
| `visit_store` | `helion.language.memory_ops.store` | `tensor.insert_slice` |
| `visit_for_loop` | `_for_loop` | `affine.for` with iter_args |
| `visit_phi` | `_phi` | `helion.phi` |
| `visit_full` | `helion.language.creation_ops.full` | `tensor.empty` + `linalg.fill` |
| `visit_get_symnode` | `_get_symnode` | `loom.get_symbol` |
| `visit_host_tensor` | `_host_tensor` | lookup in `host_tensors` map |
| `visit_subscript` | subscript ops | `tensor.extract_slice` / `tensor.expand_shape` |
| `visit_aten_full` | `aten.full.default` | `tensor.empty` + `linalg.fill` |
| `visit_aten_compute` | `aten.*` | delegates to `torch_mlir_helper` |

**ATen Operations**: When `visit_call_function` encounters an ATen op (like `aten.addmm`, `aten.bmm`, `aten.exp2`), it delegates to `torch_mlir_helper.import_aten_node_to_mlir()` and inlines the result.

### 3. `lowering_context.py` — State Management

**Purpose**: Holds all state during FX-to-MLIR conversion.

**Key Data**:
```python
@dataclass
class LoweringContext:
    builder: MLIRBuilder          # MLIR text emission
    kernel_name: str              # Function name
    bound_kernel: Any             # Source BoundKernel
    device_ir: Any                # DeviceIR
    loop_extents: dict[str, int]  # Name → total extent

    # Loop information
    outer_loops: list[LoopInfo]      # Parallel loops (tile_m, tile_n)
    reduction_loops: list[LoopInfo]  # Reduction loops (tile_k)
    block_sizes: dict[int, Any]      # BlockSizeInfo by block_id
    parallel_block_ids: list[int]    # Grid block IDs

    # Kernel arguments
    kernel_args: list[KernelArgInfo]  # Argument metadata

    # SSA mappings
    symbols: dict[str, str]       # Symbol name → SSA
    host_tensors: dict[str, str]  # Tensor name → SSA (function args)
```

**Supporting Dataclasses**:
- `LoopInfo`: block_id, name, tile_size, trip_count, iv_name
- `KernelArgInfo`: name, index, is_tensor, dtype, shape, mlir_type, ssa_name

### 4. `mlir_builder.py` — Text Emission

**Purpose**: Low-level MLIR text construction with indentation and SSA naming.

**Key Methods**:
- `emit(text)` — emit with indentation
- `push()` / `pop()` — manage nesting
- `fresh(hint)` — generate unique SSA names
- `emit_module_start(attrs)` / `emit_module_end()`
- `emit_func_start(name, args)` / `emit_func_end()`

**Utility Functions**:
- `torch_dtype_to_mlir_element_type(dtype)` → `"f32"`, `"f16"`, etc.
- `format_tensor_type(shape, element_type)` → `"tensor<?x?xf32>"`
- `is_concrete_size(size)` → True for int, False for SymInt/AutoSize

### 5. `torch_mlir_helper.py` — ATen Integration

**Purpose**: Converts ATen operations to MLIR via torch-mlir's FxImporter.

**Key Components**:
- `TorchMLIRNodeImporter`: Wraps FxImporter to import individual nodes
- `import_aten_node_to_mlir(node)`: Main entry for ATen conversion
- `inline_torch_mlir_output(mlir_text, operands, builder)`: Extracts ops from torch-mlir module and inlines into current builder with SSA renaming

**Output**: Torch dialect ops (`torch.aten.*`) that can optionally be lowered further via torch-mlir pipelines.

---

## Device IR Structure

Helion compiles kernels into an FX-based Device IR containing multiple graphs:

| Graph Type | Purpose | `block_ids` |
|------------|---------|-------------|
| `ForLoopGraphInfo` | Innermost loop body (reduction) | `[2]` (e.g., tile_k) |
| `RootGraphInfo` | Outer parallel structure | `None` |

### Device IR Nodes Reference

| FX Target | Description | MLIR Output |
|-----------|-------------|-------------|
| `helion.language.memory_ops.load` | Tile load | `tensor.extract_slice` |
| `helion.language.memory_ops.store` | Tile store | `tensor.insert_slice` |
| `helion.language._tracing_ops._host_tensor` | Kernel arg tensor | SSA from `host_tensors` |
| `helion.language._tracing_ops._phi` | Loop-carried value | `helion.phi` |
| `helion.language._tracing_ops._get_symnode` | Symbolic size | `loom.get_symbol` |
| `helion.language._tracing_ops._for_loop` | Reduction loop | `affine.for` |
| `helion.language.creation_ops.full` | Tensor init | `tensor.empty` + `linalg.fill` |
| `aten.*` | PyTorch ops | `torch.aten.*` (via torch-mlir) |

---

## Lowering Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. INPUT: BoundKernel                                               │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 2. CREATE CONTEXT: LoweringContext.from_bound_kernel()              │
│    - Extract kernel args, loop info, block sizes                    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 3. EMIT MODULE & FUNCTION: MLIRBuilder                              │
│    - Module with block size attributes                              │
│    - Function with tensor arguments                                 │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 4. WALK GRAPHS: IRVisitor.visit_graph()                             │
│    For each FX node:                                                │
│    ├─ Helion ops → visit_* methods (emit standard MLIR)            │
│    └─ ATen ops → torch_mlir_helper (emit torch dialect)            │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 5. OUTPUT: MLIR Text                                                │
│    - Dialects: affine, tensor, linalg, arith, torch, helion, loom   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Op Mapping Reference

| Device IR Node | Generated MLIR |
|----------------|----------------|
| `aten.addmm` | `torch.aten.addmm` |
| `aten.bmm` | `torch.aten.bmm` |
| `aten.exp2` | `torch.aten.exp2` |
| `aten.div.Tensor` | `torch.aten.div.Tensor` |
| `aten.amax` | `torch.aten.amax` |
| `aten.sum.dim_IntList` | `torch.aten.sum.dim_IntList` |
| `load` | `tensor.extract_slice` |
| `store` | `tensor.insert_slice` |
| `_for_loop` | `affine.for` with iter_args |
| `_phi` | `helion.phi` |
| `full` | `tensor.empty` + `linalg.fill` |
| `subscript` | `tensor.extract_slice` / `tensor.expand_shape` |
