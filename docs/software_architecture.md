# Helion FX-to-MLIR Software Architecture

This document describes the software architecture of the `helion_fx_mlir` package, which converts Helion Device IR (FX graphs) to MLIR text representation.

## Overview

The `helion_fx_mlir` package uses an **instruction-driven** approach that walks Device IR FX graphs node-by-node, generating corresponding MLIR operations. This creates a clear 1:1 correspondence between Device IR operations and MLIR output.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        BoundKernel (Helion)                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │
│  │ fake_args   │  │    env      │  │      host_function          │  │
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
                    │               │               │
                    └───────────────┼───────────────┘
                                    ▼
                            ┌───────────────┐
                            │  MLIR Text    │
                            │   Output      │
                            └───────────────┘
```

---

## Package Structure

```
src/helion_fx_mlir/
├── __init__.py              # Public API exports
├── helion_mlir.py           # Main entry point: generate_mlir()
├── ir_visitor.py            # IRVisitor: walks FX graphs node-by-node
├── mlir_builder.py          # MLIR text emission utilities
└── lowering_context.py      # Lowering state management
```

---

## Core Modules

### 1. `helion_mlir.py` - Main Entry Point

**Purpose**: Entry point for MLIR generation.

**Key Functions**:

| Function | Description |
|----------|-------------|
| `generate_mlir(bound_kernel, kernel_name)` | Generate MLIR from a bound kernel |
| `validate_with_helion_opt(mlir_text, ...)` | Validate MLIR using helion-opt or mlir-opt |

---

### 2. `ir_visitor.py` - Graph Walker

**Purpose**: Walks Device IR FX graphs node-by-node, dispatching to handlers for each op type.

#### Class: `IRVisitor`

**Responsibilities**:
- Visit all nodes in Device IR graphs in order
- Dispatch to appropriate handler based on node target
- Track SSA values for each FX node
- Handle nested graphs (ForLoopGraphInfo) recursively

**Core Methods**:

| Method | Description |
|--------|-------------|
| `visit_graph(graph_info)` | Visit all nodes in a graph |
| `visit_node(node)` | Dispatch to appropriate handler |
| `register_graph(id, info)` | Register nested graph for later visitation |

**Op Handlers**:

| Method | Device IR Target | MLIR Output |
|--------|------------------|-------------|
| `visit_get_symnode` | `_get_symnode` | `loom.get_symbol` |
| `visit_full` | `full` | `helion.full` |
| `visit_for_loop` | `_for_loop` | `affine.for` |
| `visit_phi` | `_phi` | `helion.phi` |
| `visit_host_tensor` | `_host_tensor` | func arg or `helion.alloc_like` |
| `visit_sym_size` | `aten.sym_size.int` | inline constant |
| `visit_load` | `load` | `helion.load` |
| `visit_store` | `store` | `helion.store` |
| `visit_aten_compute` | `aten.*` | `helion.call_torch` |

**Internal State**:

| Field | Type | Description |
|-------|------|-------------|
| `node_values` | `dict[str, str]` | FX node name -> MLIR SSA value |
| `graphs` | `dict[int, GraphInfo]` | Graph ID -> GraphInfo for nested graphs |
| `loop_iter_args` | `dict[str, str]` | Placeholder name -> iter_arg SSA |
| `current_loop_result` | `str | None` | Current loop's yield value |

---

### 3. `mlir_builder.py` - MLIR Text Emission

**Purpose**: Manages MLIR text generation with indentation and SSA naming.

#### Class: `MLIRBuilder`

**Core Methods**:

| Method | Description |
|--------|-------------|
| `emit(text)` | Emit a line with current indentation |
| `push()` / `pop()` | Increase/decrease indentation |
| `fresh(hint)` | Generate unique SSA name like `%hint0` |
| `build()` | Return complete MLIR text |
| `emit_module_start()` / `emit_module_end()` | Module delimiters |
| `emit_func_start(name, args, result_type)` | Function header |
| `emit_func_end()` | Function closing |

#### Utility Functions

| Function | Description |
|----------|-------------|
| `format_tensor_type(shape, element_type)` | Format `tensor<...>` type |
| `format_shape_attr(shape)` | Format shape as `[dim0, dim1, ...]` |
| `format_string_attr(value)` | Format string as `"value"` |
| `format_attr_dict(attrs)` | Format `{key = value, ...}` |

---

### 4. `lowering_context.py` - State Management

**Purpose**: Holds all state needed during the lowering process.

#### Class: `LoweringContext`

**Key Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `builder` | `MLIRBuilder` | MLIR text builder |
| `kernel_args` | `list[KernelArgInfo]` | Kernel function arguments |
| `outer_loops` | `list[LoopInfo]` | Parallel loop information |
| `reduction_loops` | `list[LoopInfo]` | Reduction loop information |
| `symbols` | `dict[str, str]` | Symbol table for `_get_symnode` |
| `host_tensors` | `dict[str, str]` | Tensor name -> function arg SSA |

**Factory Method**:

```python
@classmethod
def from_bound_kernel(cls, bound_kernel, kernel_name) -> LoweringContext
```

---

## Data Flow

### Complete Lowering Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. INPUT: BoundKernel                                               │
│    - device_ir.graphs: [ForLoopGraphInfo, RootGraphInfo]           │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 2. CREATE CONTEXT: LoweringContext.from_bound_kernel()              │
│    - Initialize MLIRBuilder                                         │
│    - Extract kernel args, loop info                                 │
│    - Set up host_tensors mapping                                    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 3. EMIT PROLOGUE                                                    │
│    builder.emit_module_start()                                      │
│    builder.emit_func_start(name, args, result_type)                │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 4. CREATE IRVisitor                                                 │
│    - Register ForLoopGraphInfo in visitor.graphs                    │
│    - visitor.visit_graph(root_graph)                                │
│                                                                     │
│    For each FX node in RootGraphInfo:                              │
│      → _get_symnode → emit loom.get_symbol                         │
│      → full → emit helion.full                                     │
│      → _for_loop → emit affine.for + visit ForLoopGraphInfo        │
│      → _phi → emit helion.phi                                      │
│      → _host_tensor → emit helion.alloc_like (for 'out')           │
│      → store → emit helion.store                                   │
│                                                                     │
│    For ForLoopGraphInfo nodes:                                      │
│      → placeholder → map to iter_args                              │
│      → _host_tensor → map to function args                         │
│      → load → emit helion.load                                     │
│      → aten.* → emit helion.call_torch                             │
│      → output → set current_loop_result for affine.yield           │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 5. EMIT EPILOGUE                                                    │
│    builder.emit("return %out : tensor<...>")                       │
│    builder.emit_func_end()                                          │
│    builder.emit_module_end()                                        │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 6. OUTPUT: builder.build() → MLIR text string                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Op Mapping Reference

| Device IR | Handler | MLIR Output | Description |
|-----------|---------|-------------|-------------|
| `_get_symnode('name')` | `visit_get_symnode` | `loom.get_symbol` | Symbolic size |
| `full([shape], val, dtype)` | `visit_full` | `helion.full` | Tensor init |
| `_for_loop(id, begin, end, args)` | `visit_for_loop` | `affine.for` | Reduction loop |
| `getitem(loop, idx)` | `visit_getitem` | (passthrough) | Loop result |
| `_phi(init, result)` | `visit_phi` | `helion.phi` | Value merge |
| `_host_tensor('out')` | `visit_host_tensor` | `helion.alloc_like` | Output alloc |
| `_host_tensor('x')` | `visit_host_tensor` | `%x` | Input reference |
| `aten.sym_size.int(t, dim)` | `visit_sym_size` | constant | Dimension |
| `load(tensor, indices)` | `visit_load` | `helion.load` | Tile load |
| `store(tensor, indices, val)` | `visit_store` | `helion.store` | Tile store |
| `aten.addmm(...)` | `visit_aten_compute` | `helion.call_torch` | ATen op |

---

## Extending the System

### Adding a New Op Handler

Add a new method to `IRVisitor`:

```python
def visit_my_op(self, node: fx.Node) -> str:
    # Get operands
    operand = self.node_values[node.args[0].name]
    
    # Emit MLIR
    result = self.builder.fresh("my_result")
    self.builder.emit(f'{result} = "my.op"({operand}) : ...')
    
    # Track SSA value
    self.node_values[node.name] = result
    return result
```

Then add dispatch in `visit_call_function`:

```python
if target is my_module.my_op:
    return self.visit_my_op(node)
```

---

## Summary

The instruction-driven architecture provides:

1. **Clear correspondence**: Each Device IR node maps to one MLIR operation
2. **Simple flow**: Walk graph nodes in order, emit corresponding MLIR
3. **Extensibility**: Add new op handlers as needed
4. **Maintainability**: Logic for each op is isolated in its handler method
