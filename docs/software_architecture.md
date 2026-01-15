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
    grid_loops: list[LoopInfo]      # Parallel loops (tile_m, tile_n)
    inner_loops: list[LoopInfo]  # Reduction loops (tile_k)
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
---

## LoweringContext and IRVisitor Data Flow

This section details the information held by `LoweringContext` and how `IRVisitor` uses it to correctly generate MLIR.

### LoweringContext Fields

`LoweringContext` is a dataclass that holds all state during FX-to-MLIR conversion. It is created once via `LoweringContext.from_bound_kernel()` before graph walking begins.

#### Core Infrastructure

| Field | Type | Description |
|-------|------|-------------|
| `builder` | `MLIRBuilder` | Shared builder for MLIR text emission. IRVisitor accesses this to emit operations via `self.builder.emit()`, `self.builder.fresh()`, etc. |
| `bound_kernel` | `BoundKernel` | The source Helion kernel containing `fake_args`, `env`, and `host_function`. |
| `device_ir` | `DeviceIR` | The Device IR containing the FX graphs to be lowered. |

#### Type Information

| Field | Type | Description |
|-------|------|-------------|
| `element_type` | `str` | MLIR element type (e.g., `"f32"`, `"f16"`). Derived from the first tensor argument's dtype. |
| `tensor_type` | `str` | Default dynamic tensor type (e.g., `"tensor<?x?xf32>"`). Used for intermediate tile tensors. |

**Usage in IRVisitor**: `tensor_type` is used extensively in:
- `visit_for_loop`: to build `iter_args` and result types
- `visit_phi`: to build the phi operation type signature
- `visit_load`/`visit_store`: as the result/value type
- `visit_subscript`: as the output tensor type
- `visit_aten_compute`: to determine output types

#### Loop Information

| Field | Type | Description |
|-------|------|-------------|
| `grid_loops` | `list[LoopInfo]` | Parallel (outer) loops. Each has `block_id`, `name`, `tile_size`, `trip_count`, `total_extent`, `is_symbolic`, `iv_name`. |
| `inner_loops` | `list[LoopInfo]` | Reduction (inner) loops. Same structure as `grid_loops`. |
| `loop_extents` | `dict[str, int]` | Maps loop name to total iteration extent (e.g., `{"tile_m": 128, "tile_n": 256}`). |
| `block_sizes` | `dict[int, BlockSizeInfo]` | Maps block_id to Helion's BlockSizeInfo containing tile size and debug names. |
| `parallel_block_ids` | `list[int]` | Block IDs for parallel (outer) loops, derived from `device_ir.grid_block_ids`. |

**Usage in IRVisitor**:
- `visit_sym_size`: Uses `grid_loops` and `inner_loops` to map dimension indices to loop induction variables:
  ```python
  if dim < len(self.ctx.grid_loops):
      block_id = self.ctx.grid_loops[dim].block_id
      iv_name = f"%iv_block{block_id}"
  ```
- This allows `sym_size_int(tensor, 0)` to resolve to `%iv_block0` (first outer loop IV).

#### SSA Value Mappings

| Field | Type | Description |
|-------|------|-------------|
| `symbols` | `dict[str, str]` | Maps symbol names to SSA values (e.g., `{"block_size_0": "%block_size_00"}`). |
| `host_tensors` | `dict[str, str]` | Maps tensor names to function argument SSA values (e.g., `{"x": "%x", "out": "%out"}`). |

**Usage in IRVisitor**:
- `visit_get_symnode`: Stores the emitted symbol SSA in `ctx.symbols`:
  ```python
  self.ctx.symbols[name] = ssa  # e.g., symbols["block_size_0"] = "%block_size_00"
  ```
- `visit_host_tensor`: Looks up tensor names in `ctx.host_tensors`:
  ```python
  ssa = self.ctx.host_tensors.get(tensor_name)  # e.g., "x" → "%x"
  if ssa is None:
      raise ValueError(f"Host tensor '{tensor_name}' not found...")
  ```

#### Kernel Arguments

| Field | Type | Description |
|-------|------|-------------|
| `kernel_args` | `list[KernelArgInfo]` | Metadata for each kernel parameter: name, index, is_tensor, dtype, shape, mlir_type, ssa_name. |

**Usage in IRVisitor**:
- `_get_tensor_type`: Searches `kernel_args` to find the MLIR type for a specific tensor:
  ```python
  for arg in self.ctx.kernel_args:
      if arg.name == tensor_name:
          return arg.mlir_type  # e.g., "tensor<128x256xf32>"
  ```

### IRVisitor's Own State

In addition to using `LoweringContext`, `IRVisitor` maintains its own transient state during graph walking:

| Field | Type | Purpose |
|-------|------|---------|
| `node_values` | `dict[str, str]` | Maps FX node names to MLIR SSA values. Grows as nodes are visited. |
| `graphs` | `dict[int, GraphInfo]` | Registered ForLoopGraphInfo objects by graph_id. Set before walking. |
| `loop_iter_args` | `dict[str, str]` | Maps placeholder names to iter_arg SSA values during loop body visitation. |
| `current_loop_result` | `str \| list[str]` | The SSA value(s) from the inner graph's output node. Used for `affine.yield`. |
| `loop_depth` | `int` | Current nesting depth for loops. |
| `initial_acc_ssa` | `dict[str, str]` | Maps accumulator node names to their initial SSA values (before loop entry). Used by `visit_phi`. |
| `block_size_ssa` | `dict[int, str]` | Pre-computed block size SSA values by block_id. |
| `reduction_trip_counts` | `dict[int, str]` | Pre-computed trip count SSA values for reduction loops. |

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LoweringContext (Shared State)                      │
│                                                                               │
│  ┌─────────────┐    ┌────────────────┐    ┌─────────────────────────────┐   │
│  │   builder   │    │  tensor_type   │    │         host_tensors        │   │
│  │ (MLIRBuilder│    │ "tensor<?x?    │    │ {"x": "%x", "out": "%out"}  │   │
│  │  emit/fresh)│    │   xf32>"       │    │                             │   │
│  └──────┬──────┘    └───────┬────────┘    └────────────┬────────────────┘   │
│         │                   │                          │                     │
│  ┌──────┴──────┐    ┌───────┴────────┐    ┌────────────┴────────────────┐   │
│  │  grid_loops │    │   symbols      │    │       kernel_args           │   │
│  │ [LoopInfo]  │    │ {"block_size_0"│    │ [KernelArgInfo for each arg]│   │
│  │             │    │  : "%bs0"}     │    │                             │   │
│  └──────┬──────┘    └───────┬────────┘    └────────────┬────────────────┘   │
└─────────┼───────────────────┼──────────────────────────┼────────────────────┘
          │                   │                          │
          ▼                   ▼                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          IRVisitor (Transient State)                         │
│                                                                               │
│  ┌─────────────────┐    ┌────────────────┐    ┌──────────────────────────┐  │
│  │   node_values   │    │ loop_iter_args │    │  reduction_trip_counts   │  │
│  │ {"load": "%s1"} │    │ {"arg0_1": ... │    │ {2: "%trip_count0"}      │  │
│  └────────┬────────┘    └───────┬────────┘    └────────────┬─────────────┘  │
│           │                     │                          │                 │
│           └─────────────────────┴──────────────────────────┘                 │
│                                    │                                         │
│                                    ▼                                         │
│                          ┌──────────────────┐                                │
│                          │   MLIR Output    │                                │
│                          │ via builder.emit │                                │
│                          └──────────────────┘                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Example: Visiting `_host_tensor("x")`

1. FX node: `call_function(_host_tensor, ('x',))`
2. IRVisitor calls `visit_host_tensor(node)`
3. Retrieves `tensor_name = "x"` from `node.args[0]`
4. Looks up in `self.ctx.host_tensors.get("x")` → returns `"%x"`
5. Stores `self.node_values["x"] = "%x"`
6. Returns `"%x"` for downstream nodes to reference

### Example: Visiting `aten.sym_size.int(tensor, 0)`

1. FX node: `call_function(aten.sym_size.int, (tensor_node, 0))`
2. IRVisitor calls `visit_sym_size(node)` with `dim = 0`
3. Checks `len(self.ctx.grid_loops)` to see if dimension is within outer loops
4. Gets `block_id = self.ctx.grid_loops[0].block_id` (e.g., `0`)
5. Returns `iv_name = "%iv_block0"` as the SSA value
6. Stores in `self.node_values` for downstream reference

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
