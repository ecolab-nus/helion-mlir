# Helion → MLIR Lowering

This repository hosts an instruction-driven lowering path that translates
Helion kernels into MLIR by walking Device IR FX graphs node-by-node. Each
Device IR operation is mapped to a corresponding MLIR operation.

## MLIR Generation

The main entry point is `generate_mlir()` which walks Device IR instruction-by-instruction:

```python
from helion_fx_mlir import generate_mlir, validate_with_helion_opt

# Generate MLIR from a bound Helion kernel
mlir_text = generate_mlir(bound_kernel, kernel_name="matmul")

# Validate with mlir-opt
result = validate_with_helion_opt(mlir_text)
```

### Op Mapping

The instruction-driven approach maps Device IR operations to MLIR:

| Device IR Operation | MLIR Output | Description |
|---------------------|-------------|-------------|
| `_get_symnode('name')` | `loom.get_symbol` | Symbolic tile size reference |
| `full([shape], val, dtype)` | `helion.full` | Tensor initialization |
| `_for_loop(id, begin, end, args)` | `affine.for` | Reduction loop with iter_args |
| `getitem(loop, idx)` | (passthrough) | Extract loop result |
| `_phi(init, result)` | `helion.phi` | Loop-carried value merge |
| `_host_tensor('name')` | func arg / `helion.alloc_like` | Tensor reference or allocation |
| `aten.sym_size.int(t, dim)` | inline constant | Tensor dimension |
| `load(tensor, indices)` | `helion.load` | Tile load |
| `store(tensor, indices, val)` | `helion.store` | Tile store |
| `aten.*` compute | `helion.call_torch` | PyTorch operation |

### Example Output

For a matmul kernel, the generated MLIR looks like:

```mlir
module attributes {loom.block_m = -1 : index, loom.block_n = -1 : index, loom.block_k = -1 : index} {
  func.func @matmul(%x: tensor<?x?xf32>, %y: tensor<?x?xf32>) -> tensor<128x256xf32> {
    %block_size_0 = "loom.get_symbol"() {name = "block_size_0"} : () -> index
    %block_size_1 = "loom.get_symbol"() {name = "block_size_1"} : () -> index
    %full = "helion.full"(%block_size_0, %block_size_1) {fill_value = 0.0, dtype = f32} : (index, index) -> tensor<?x?xf32>
    %x_size1 = "loom.get_symbol"() {name = "x_size1"} : () -> index
    %for_result = affine.for %iv = 0 to %x_size1 iter_args(%acc = %full) -> (tensor<?x?xf32>) {
      %load = "helion.load"(%x) {indices = [...], fx_node = "load"} : (tensor<?x?xf32>) -> tensor<?x?xf32>
      %load_1 = "helion.load"(%y) {indices = [...], fx_node = "load_1"} : (tensor<?x?xf32>) -> tensor<?x?xf32>
      %acc_new = "helion.call_torch"(%acc, %load, %load_1) {fn_name = "aten.addmm"} : (...) -> tensor<?x?xf32>
      affine.yield %acc_new : tensor<?x?xf32>
    }
    %phi = "helion.phi"(%full, %for_result) {fx_node = "_phi"} : (...) -> tensor<?x?xf32>
    %out = "helion.alloc_like"(%x) {shape = [128, 256]} : (tensor<?x?xf32>) -> tensor<128x256xf32>
    "helion.store"(%out, %phi) {indices = [...], fx_node = "store"} : (...) -> ()
    return %out : tensor<128x256xf32>
  }
}
```

## Device IR Structure

Helion compiles kernels into an FX-based Device IR containing multiple graphs:

| Graph Type | Purpose | `block_ids` |
|------------|---------|-------------|
| `ForLoopGraphInfo` | Innermost loop body (reduction loops) | `[2]` (e.g., tile_k) |
| `RootGraphInfo` | Outer parallel structure and control flow | `None` |

### Device IR Nodes

Key FX targets and their meaning:

| FX Target | Description | Example Node Name |
|-----------|-------------|-------------------|
| `helion.language.memory_ops.load` | Tile load from tensor | `load`, `load_1` |
| `helion.language.memory_ops.store` | Tile store to tensor | `store` |
| `helion.language._tracing_ops._host_tensor` | Reference to kernel argument tensor | `x`, `y`, `out` |
| `helion.language._tracing_ops._phi` | Loop-carried value (reduction) | `_phi` |
| `helion.language._tracing_ops._get_symnode` | Symbolic size reference | `block_size_0` |
| `helion.language._tracing_ops._for_loop` | Reduction loop marker | `_for_loop` |
| `helion.language.creation_ops.full` | Tensor initialization | `acc` |
| `aten.addmm.default` | Matrix multiply-add | `acc` |

### Original Python to Device IR Mapping

**Original Python Kernel:**
```python
@helion.kernel
def matmul(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> None:
    for tile_m, tile_n in hl.tile([x.size(0), y.size(1)]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(x.size(1)):
            x_tile = x[tile_m, tile_k]  # load
            y_tile = y[tile_k, tile_n]  # load_1
            acc = torch.addmm(acc, x_tile, y_tile)  # acc
        out[tile_m, tile_n] = acc  # store
```

**Device IR (ForLoopGraphInfo)** - Inner `tile_k` loop:
```
call_function  x               _host_tensor    ('x',)
call_function  load            load            (x, [sym_size_int, block_size_2], ...)
call_function  y               _host_tensor    ('y',)
call_function  load_1          load            (y, [block_size_2, sym_size_int_1], ...)
call_function  acc             aten.addmm      (_new_var, load, load_1)
```

**Device IR (RootGraphInfo)** - Outer control flow:
```
call_function  block_size_0    _get_symnode    ('block_size_0',)
call_function  block_size_1    _get_symnode    ('block_size_1',)
call_function  acc             full            ([block_size_0, block_size_1], 0.0, ...)
call_function  _for_loop       _for_loop       (0, [0], [x_size1], [acc])
call_function  _phi            _phi            (acc, getitem)
call_function  out             _host_tensor    ('out',)
call_function  store           store           (out, [...], _phi, ...)
```

## Architecture

The lowering architecture consists of:

- **`IRVisitor`**: Walks FX graphs instruction-by-instruction, dispatching to handlers for each op type
- **`LoweringContext`**: Holds state during lowering (SSA values, symbol table, kernel args)
- **`MLIRBuilder`**: Handles MLIR text emission and SSA naming

## Limitations

> [!IMPORTANT]
> Current limitations:

- **Single ForLoopGraphInfo**: Only one nested loop is supported
- **Simple indexing**: Only `x[tile_m, tile_k]` style indexing is supported
- **Symbolic shapes only**: Dynamic runtime shapes are not supported

## Building `helion-opt`

The Helion dialect and driver live under `mlir/`. Build with:

```bash
cmake -S . -B build
cmake --build build --target helion-opt
```

This produces `build/mlir/helion-opt`. Note that new ops like `helion.full` and
`loom.get_symbol` require using `mlir-opt -allow-unregistered-dialect` until
they are registered in the C++ dialect.

## Running Examples

```bash
python examples/matmul.py
```

This prints the Device IR and generated MLIR, then validates with `mlir-opt`.
