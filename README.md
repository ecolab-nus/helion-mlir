# Helion → MLIR Lowering

This repository provides an instruction-driven lowering path that translates
[Helion](https://github.com/pytorch-labs/helion) kernels into MLIR. It walks Device IR FX graphs node-by-node,
mapping each operation to corresponding MLIR dialects.

For detailed architecture, FX graph structure, and lowering internals, see [Software Architecture](docs/software_architecture.md).

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

This project requires **Python 3.11**.

1. **Create and activate a virtual environment**:
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   **Note**: `torch-mlir` is required for ATen operation lowering and will be installed via `requirements.txt`.

## Current Limitations

- **Masking**: `_mask_to` passes through tensors without boundary checks
- **Dynamic Shapes**: Full dynamic shape support is work-in-progress

## License

MIT License. See [LICENSE](LICENSE).
