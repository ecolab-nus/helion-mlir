# Helion → MLIR Lowering

This repository provides a tool to lower [Helion](https://github.com/helion-project/helion) kernels into MLIR (Affine + Linalg-on-Tensors). It reuses Helion's front-end, starting from Helion's Device IR FX graphs and translating them into MLIR.

## Motivation

Helion lowers to Triton, which is too low-level for our purposes. We want a representation with:
- **Symbolic shapes** for flexible dimension handling
- **High-level compute** (`linalg.generic`, `linalg.matmul`, etc.)
- **Structured control flow** (`affine.for`, `affine.parallel`)
- **Value semantics** (`tensor`, not `memref`)

This higher-level representation enables analyses that can target diverse architectures.

## Output

The generated MLIR includes custom `loom.*` operations to support symbolic shapes (which MLIR does not natively support). This output is designed to be consumed by downstream compiler infrastructure for targeting various architectures, especially dataflow architectures.



## Quick Start

```python
from helion_mlir import generate_mlir, validate_with_mlir_opt

# Generate MLIR from a bound Helion kernel
mlir_text = generate_mlir(bound_kernel, kernel_name="matmul")

# Validate with mlir-opt
result = validate_with_mlir_opt(mlir_text)
```

## Op Mapping Overview

The system maps operations from two sources:

| Operation Type | Examples | Generated Dialect |
|----------------|----------|-------------------|
| **Control Flow** | `_for_loop` | `affine` (`affine.for`, `affine.parallel`) |
| **Memory** | `load`, `store`, `subscript` | `tensor` (`extract_slice`, `insert_slice`, `expand_shape`) |
| **Tensor Creation** | `full`, `zeros` | `tensor.empty` + `linalg.fill` |
| **Compute** | `addmm`, `bmm`, `exp2`, `amax`, ... | `linalg` (via torch-mlir lowering) |
| **Symbols** | `_get_symnode` | `loom.get_symbol` |

### Helion-Specific Operations

| Device IR Node | Generated MLIR |
|----------------|----------------|
| `_for_loop` | `affine.for` with `iter_args` |
| `_phi` | Loop result SSA (simplified merge pattern) |
| `load` | `tensor.extract_slice` (tile sizes from FakeTensor metadata) |
| `store` | `tensor.insert_slice` (tile sizes from FakeTensor metadata) |
| `subscript` | `tensor.extract_slice` / `tensor.expand_shape` |
| `full` / `zeros` | `tensor.empty` + `linalg.fill` |
| `_host_tensor` | SSA lookup (function parameters) |
| `_mask_to` | Pass-through (TODO: boundary checks) |

### ATen Operations

All ATen operations (`aten.*`) are lowered through **torch-mlir** integration. The generated MLIR uses **linalg-on-tensors** dialect:

```
aten.addmm   → linalg.generic (matmul pattern)
aten.bmm     → linalg.batch_matmul
aten.exp2    → linalg.generic (math.powf)
aten.amax    → linalg.generic (reduction)
aten.sum     → linalg.generic (reduction)
...
```

> **Note**: torch-mlir first imports ops as `torch.aten.*`, then immediately lowers them to `linalg-on-tensors`. The final output contains only `linalg.*` operations.

See [`torch_mlir_helper.py`](src/helion_mlir/torch_mlir_helper.py) for the FxImporter-based integration.

## Package Structure

```
src/helion_mlir/
├── __init__.py              # Public API exports
├── helion_mlir.py           # Entry point: generate_mlir()
├── ir_visitor.py            # IRVisitor: walks FX graphs, dispatches to visit_* methods
├── lowering_context.py      # LoweringContext: state (loops, args, SSA mappings)
├── mlir_utils.py            # MLIROutputHelper: text emission, SSA naming, indentation
├── torch_mlir_helper.py     # torch-mlir integration for ATen ops
└── debug_utils.py           # Debug utilities and MLIR validation
```

For detailed architecture, FX graph structure, and lowering internals, see [Software Architecture](docs/software_architecture.md).
For maintainer-facing ownership boundaries in the refactored lowering pipeline, see [Lowering Pipeline And Ownership](docs/lowering_pipeline_and_ownership.md).
For expression-based block-loop bound lowering details (including `mamba_chunk_scan`), see [Block Loop Bounds Lowering README](docs/block_loop_bounds_lowering_readme.md).


## Validation

MLIR validation uses `mlir-opt` with `-allow-unregistered-dialect` to handle:
- `loom.*` operations (symbolic block sizes)

```bash
mlir-opt -allow-unregistered-dialect output.mlir
```

The `validate_with_mlir_opt()` function (in `debug_utils.py`) automates this by searching for `mlir-opt` in common locations.

## Running Examples

### Matrix Multiplication

```bash
python examples/matmul.py
```

Prints Device IR, generated MLIR, and validates with `mlir-opt`.

### Flash Attention

```bash
python examples/attn.py
```

Demonstrates a more complex kernel with 3D tensors, batch matrix operations, and reduction loops.

## Environment Preparation

This project requires **Python 3.10+**.

1. **Create and activate a virtual environment**:
   ```bash
   python3.10 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   python -m pip install -r requirements.txt
   ```

   **Note**: `requirements.txt` installs this project in editable mode, resolves
   `torch-mlir` from its dev-wheel source, and selects CPU-only PyTorch wheels.

## Current Limitations

- **Masking**: `_mask_to` passes through tensors without boundary checks
- **Dynamic Shapes**: Full dynamic shape support is work-in-progress

## Internal Architecture

The lowering flow is now structured around four explicit internal stages:

1. `build_kernel_analysis()` gathers immutable facts from `bound_kernel`.
2. `LoweringSession` owns mutable lowering state.
3. `ModuleEmitter` emits module/function scaffolding and precomputed symbols.
4. `IRVisitor` walks FX graphs using domain-grouped handler registration under `src/helion_mlir/handlers/`.

## License

MIT License. See [LICENSE](LICENSE).
