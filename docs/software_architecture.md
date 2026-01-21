# Helion FX-to-MLIR Software Architecture

This document describes the software architecture of the `helion_mlir` package, which converts Helion Device IR (FX graphs) to MLIR text representation.

## Overview

The `helion_mlir` package uses a **visitor pattern** to walk Device IR FX graphs node-by-node. It generates:
1. **Standard MLIR dialects** (`tensor`, `affine`, `linalg`, `arith`) for Helion-specific operations
2. **Linalg dialect** (`linalg.generic`, `linalg.fill`, etc.) via **torch-mlir** for ATen operations

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
            ┌───────────┐   ┌──────────────┐   ┌────────────────┐
            │ IRVisitor │   │LoweringContext│  │MLIROutputHelper│
            │  (Walk)   │   │   (State)     │  │  (Emission)    │
            └───────────┘   └──────────────┘   └────────────────┘
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
src/helion_mlir/
├── __init__.py              # Public API exports
├── helion_mlir.py           # Main entry: generate_mlir()
├── ir_visitor.py            # IRVisitor: walks FX graphs, dispatches to visit_* methods
├── lowering_context.py      # LoweringContext: state (loops, args, SSA mappings)
├── mlir_utils.py            # MLIROutputHelper: text emission, SSA naming, indentation
├── torch_mlir_helper.py     # torch-mlir integration for ATen ops
└── debug_utils.py           # Debug utilities, MLIR validation (validate_with_mlir_opt)
```

---

## Core Modules

### 1. `helion_mlir.py` — Entry Point

**Purpose**: Main entry point for MLIR generation.

**Key Functions**:
- `generate_mlir(bound_kernel, kernel_name)` → MLIR text
- `_collect_host_tensor_names()` → pre-scans graphs for tensor function parameters

### 2. `ir_visitor.py` — Graph Walker

**Purpose**: Walks Device IR FX graphs and generates MLIR via `visit_*` methods.

**Key Methods**:
| Method | Device IR Target | Generated MLIR |
|--------|------------------|----------------|
| `visit_load` | `helion.language.memory_ops.load` | `tensor.extract_slice` (sizes from FakeTensor metadata via Origin) |
| `visit_store` | `helion.language.memory_ops.store` | `tensor.insert_slice` (sizes from FakeTensor metadata via Origin) |
| `visit_for_loop` | `_for_loop` | `affine.for` with iter_args |
| `visit_phi` | `_phi` | Loop result SSA (simplified merge pattern detection) |
| `visit_full` | `helion.language.creation_ops.full` | `tensor.empty` + `linalg.fill` |
| `visit_get_symnode` | `_get_symnode` | SSA lookup via Origin (BlockSizeOrigin → `block_size_ssa`, etc.) |
| `visit_host_tensor` | `_host_tensor` | lookup in `host_tensors` map |
| `visit_subscript` | subscript ops | `tensor.extract_slice` / `tensor.expand_shape` |
| `visit_aten_full` | `aten.full.default` | `tensor.empty` + `linalg.fill` |
| `visit_aten_compute` | `aten.*` | delegates to `torch_mlir_helper` |

**ATen Operations**: When `visit_call_function` encounters an ATen op (like `aten.addmm`, `aten.bmm`, `aten.exp2`), it delegates to `torch_mlir_helper.import_aten_node_to_mlir()` and inlines the result.

### 3. `lowering_context.py` — State Management

**Purpose**: Holds all state during FX-to-MLIR conversion.

**Key Data**:
```python
class LoweringContext:
    builder: MLIRBuilder          # MLIR text emission
    kernel_name: str              # Function name
    bound_kernel: Any             # Source BoundKernel
    
    # Type mappings
    arg_mlir_types: dict[str, str]  # Kernel arg name → MLIR type string (tensors only)
    
    # Loop information
    loop_extents: dict[int, int]    # block_id → total extent

    # SSA mappings
    symbols: dict[str, str]       # Symbol name → SSA
    host_tensors: dict[str, str]  # Tensor name → SSA (function args)
    
    # Pre-computed MLIR values
    block_size_ssa: dict[int, str]   # Block ID → block size SSA value
    reduction_trip_counts: dict[int, str]  # Block ID → trip count SSA
    
    # SSA Value Tracking (populated during graph walking)
    node_values: dict[str, str]      # FX node name → MLIR SSA value
    node_types: dict[str, str]       # FX node name → MLIR type string
    initial_acc_ssa: dict[str, str]  # Accumulator node → initial SSA (for phi)
```

**Property Methods**:
- `device_ir`: Get DeviceIR from bound_kernel
- `block_sizes`: Get block sizes as dict from bound_kernel.env
- `parallel_block_ids`: Get grid block IDs for parallel loops
- `env`: Get CompileEnvironment from bound_kernel

### 4. `mlir_utils.py` — Text Emission

`mlir_utils.py` provides the `MLIROutputHelper` class, which handles the string construction of the MLIR, including indentation and SSA naming.

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
- `import_aten_node_to_mlir(node)`: Main entry for ATen conversion, returns linalg-on-tensors MLIR
- `inline_torch_mlir_output(mlir_text, operands, builder)`: Extracts ops from torch-mlir module and inlines into current builder with SSA renaming

**Output**: Linalg-on-tensors ops (`linalg.generic`, `linalg.fill`, `linalg.matmul`, etc.) that integrate directly with the tensor dialect.

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
| `helion.language._tracing_ops._phi` | Loop-carried value | Loop result SSA (simplified merge) |
| `helion.language._tracing_ops._get_symnode` | Symbolic size | SSA lookup via Origin type |
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
│ 2. CREATE CONTEXT: LoweringContext(bound_kernel)                    │
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
│ 4. PRE-EMIT BLOCK SIZES: loom.get_symbol for each block_id         │
│    - Store SSA values in ctx.block_size_ssa                         │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 5. WALK GRAPHS: IRVisitor.visit_graph()                             │
│    For each FX node:                                                │
│    ├─ Helion ops → visit_* methods (emit standard MLIR)            │
│    └─ ATen ops → torch_mlir_helper (emit torch dialect)            │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 6. OUTPUT: MLIR Text                                                │
│    - Dialects: affine, tensor, linalg, arith, torch, loom           │
└─────────────────────────────────────────────────────────────────────┘
```

## LoweringContext and IRVisitor Data Flow

This section details the information held by `LoweringContext` and how `IRVisitor` uses it to correctly generate MLIR.

### LoweringContext Fields

`LoweringContext` is a class that holds all state during FX-to-MLIR conversion. It is created once via `LoweringContext(bound_kernel)` before graph walking begins.

#### Core Infrastructure

| Field | Type | Description |
|-------|------|-------------|
| `builder` | `MLIROutputHelper` | Shared builder for MLIR text emission. IRVisitor accesses this to emit operations via `self.ctx.mlir_output_helper.emit()`, `self.ctx.mlir_output_helper.fresh()`, etc. |
| `bound_kernel` | `BoundKernel` | The source Helion kernel containing `fake_args`, `env`, and `host_function`. |

#### Type Information

| Field | Type | Description |
|-------|------|-------------|
| `arg_mlir_types` | `dict[str, str]` | Maps kernel argument name to MLIR type string (e.g., `{"x": "tensor<128x256xf32>"}`). Only tensor args are included. |

#### Loop Information

| Field | Type | Description |
|-------|------|-------------|
| `loop_extents` | `dict[int, int]` | Maps block_id to total iteration extent (e.g., `{0: 128, 1: 256, 2: 128}`). |

**Property Methods**:
- `parallel_block_ids`: Returns list of block IDs for parallel (outer) loops from `device_ir.grid_block_ids`.
- `block_sizes`: Returns dict mapping block_id → BlockSizeInfo from `env.block_sizes`.

#### SSA Value Mappings

| Field | Type | Description |
|-------|------|-------------|
| `symbols` | `dict[str, str]` | Maps symbol names to SSA values (e.g., `{"block_size_0": "%block_size_00"}`). |
| `host_tensors` | `dict[str, str]` | Maps tensor names to function argument SSA values (e.g., `{"x": "%x", "out": "%out"}`). |
| `block_size_ssa` | `dict[int, str]` | Pre-computed block size SSA values by block_id. |
| `reduction_trip_counts` | `dict[int, str]` | Pre-computed trip count SSA values for reduction loops. |

**Usage in IRVisitor**:
- `visit_get_symnode`: Uses Origin-based lookup:
  - `BlockSizeOrigin` → Use pre-emitted `ctx.block_size_ssa[block_id]`
  - Other origins → Use `shape_env.var_to_val` for concrete value → `arith.constant`
- `visit_host_tensor`: Looks up tensor names in `ctx.host_tensors`:
  ```python
  ssa = self.ctx.host_tensors.get(tensor_name)  # e.g., "x" → "%x"
  ```

### IRVisitor's Own State

In addition to using `LoweringContext`, `IRVisitor` accesses state stored in `ctx` during graph walking:

| Field | Type | Purpose |
|-------|------|---------|
| `ctx.node_values` | `dict[str, str]` | Maps FX node names to MLIR SSA values. Grows as nodes are visited. |
| `ctx.node_types` | `dict[str, str]` | Maps FX node names to MLIR type strings. |
| `ctx.graphs` | `dict[int, GraphInfo]` | Registered ForLoopGraphInfo objects by graph_id. Set before walking. |
| `ctx.initial_acc_ssa` | `dict[str, str]` | Maps accumulator node names to their initial SSA values (before loop entry). |

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LoweringContext (Shared State)                      │
│                                                                               │
│  ┌─────────────┐    ┌────────────────┐    ┌─────────────────────────────┐   │
│  │   builder   │    │ arg_mlir_types │    │         host_tensors        │   │
│  │ (MLIRBuilder│    │ {"x": "tensor  │    │ {"x": "%x", "out": "%out"}  │   │
│  │  emit/fresh)│    │   <128x256xf32>│    │                             │   │
│  └──────┬──────┘    │   "}           │    └────────────┬────────────────┘   │
│         │           └───────┬────────┘                 │                     │
│  ┌──────┴──────┐    ┌───────┴────────┐    ┌────────────┴────────────────┐   │
│  │loop_extents │    │ block_size_ssa │    │  reduction_trip_counts      │   │
│  │ {0:128,..}  │    │ {0: "%bs0"}    │    │ {2: "%trip_count0"}         │   │
│  │             │    │                │    │                             │   │
│  └──────┬──────┘    └───────┬────────┘    └────────────┬────────────────┘   │
└─────────┼───────────────────┼──────────────────────────┼────────────────────┘
          │                   │                          │
          ▼                   ▼                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          IRVisitor (using ctx state)                          │
│                                                                               │
│  ┌─────────────────┐    ┌────────────────┐    ┌──────────────────────────┐  │
│  │ ctx.node_values │    │ ctx.node_types │    │   ctx.initial_acc_ssa    │  │
│  │ {"load": "%s1"} │    │ {"load": ...}  │    │ {acc_name: "%init_ssa"}  │  │
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
5. Stores `self.ctx.node_values["x"] = "%x"`
6. Returns `"%x"` for downstream nodes to reference

### Example: Visiting `aten.sym_size.int(tensor, 0)`

1. FX node: `call_function(aten.sym_size.int, (tensor_node, 0))`
2. IRVisitor calls `visit_sym_size(node)` with `dim = 0`
3. Resolves tensor SSA from `ctx.node_values`
4. Emits `tensor.dim %tensor_ssa, %dim_idx : tensor_type` to get the dimension size
5. Returns the dimension size SSA value
6. Stores in `self.ctx.node_values` for downstream reference

### Example: Visiting `load` with Origin-Based Tile Size Resolution

1. FX node: `call_function(load, (tensor_node, indices))`
2. IRVisitor calls `visit_load(node)`
3. Gets tile sizes from `node.meta['val'].size()` (FakeTensor metadata)
4. For each dimension's SymInt:
   - Uses `Origin.get()` to determine origin type
   - `BlockSizeOrigin` → lookup `ctx.block_size_ssa[block_id]`
   - Other origin → emit `arith.constant` or lookup existing SSA
5. Emits `tensor.extract_slice` with computed offsets, sizes, strides
6. Stores result SSA in `ctx.node_values`

### Example: Visiting `_phi` with Simplified Merge Pattern

1. FX node: `call_function(_phi, (initial_val, getitem_node))`
2. IRVisitor calls `visit_phi(node)`
3. Detects loop merge pattern:
   - `rhs_node.target == operator.getitem`
   - `rhs_node.args[0]` is a `_for_loop` node
4. Uses the loop result SSA directly: `%result#i` from the corresponding `affine.for`
5. No separate `helion.phi` operation emitted
6. Stores in `ctx.node_values`

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
| `_phi` | Loop result SSA (simplified) |
| `full` | `tensor.empty` + `linalg.fill` |
| `subscript` | `tensor.extract_slice` / `tensor.expand_shape` |
