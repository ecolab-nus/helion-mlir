# Helion → MLIR Lowering Policy

This repository currently hosts an experimental, Python-based lowering path that
translates Helion kernels into a textual MLIR "stage-0" module. The goal is to
capture the loop structure, data movement, and FX provenance needed for further
passes while iterating toward richer MLIR integrations. The project now ships a
bespoke `helion` dialect plus a standalone `helion-opt` driver so the emitted IR
parses without depending on `-allow-unregistered-dialect`.

## Tile Size Handling

The MLIR generator supports both **concrete** and **symbolic** tile sizes based on
Helion's `static_shapes` setting:

| `static_shapes` | Tile Size Type | MLIR Representation |
|-----------------|----------------|---------------------|
| `True` | Concrete `int` | Emitted as constants: `arith.constant 128 : index` |
| `False` | `SymInt` | Emitted as function arguments: `%tile_m_size: index` |

Both modes use the **affine dialect** (`affine.parallel`, `affine.for`, `affine.min`, 
`affine.apply`). Symbolic tile sizes are passed as affine symbols, enabling proper
affine analysis while maintaining dynamic behavior:

- Trip counts: `affine.apply affine_map<()[s0, s1] -> (s0 ceildiv s1)>()[%dim, %tile_size]`
- Tile sizes: `affine.min affine_map<(d0)[s0, s1] -> (s1, s0 - d0 * s1)>(%iv)[%dim, %tile_size]`

Example function signatures:

```mlir
// static_shapes=True (concrete tile sizes)
func.func @matmul(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32>

// static_shapes=False (symbolic tile sizes as function arguments)
func.func @matmul(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, 
                  %tile_m_size: index, %tile_n_size: index, %tile_k_size: index) -> tensor<?x?xf32>
```

## FX Graph Structure

Helion compiles kernels into an FX-based Device IR containing multiple graphs:

| Graph Type | Purpose | `block_ids` |
|------------|---------|-------------|
| `ForLoopGraphInfo` | Innermost loop body (reduction loops) | `[2]` (e.g., tile_k) |
| `RootGraphInfo` | Outer parallel structure and control flow | `None` |

### FX Node Types

FX nodes represent operations in the Helion kernel. Each node has:
- `op`: Operation type (`call_function`, `placeholder`, `output`)
- `name`: SSA-style name used in MLIR (`fx_node` attribute)
- `target`: The Python function being called
- `args`: Arguments (often other FX nodes)

**Key FX targets and their meaning:**

| FX Target | Description | Example Node Name |
|-----------|-------------|-------------------|
| `helion.language.memory_ops.load` | Tile load from tensor | `load`, `load_1` |
| `helion.language.memory_ops.store` | Tile store to tensor | `store` |
| `helion.language._tracing_ops._host_tensor` | Reference to kernel argument tensor | `x`, `y`, `out` |
| `helion.language._tracing_ops._phi` | Loop-carried value (reduction) | `_phi` |
| `aten.addmm.default` | Matrix multiply-add | `acc` |
| `helion.language._tracing_ops._for_loop` | Reduction loop marker | `_for_loop` |

## Loop & Tensor Mapping

| Helion construct | FX representation | MLIR lowering | Notes |
| ---------------- | ----------------- | ------------- | ----- |
| `hl.tile(...)` outer blocks | `_for_loop` in RootGraph | `affine.parallel` | Trip counts via `affine.apply` with ceildiv |
| Nested `hl.tile` / reduction | ForLoopGraphInfo with `block_ids` | `affine.for` with `iter_args` | Each block_id becomes a loop |
| Loop-carried values | `_phi` FX nodes | `helion.phi` | Captures initial and final values |
| Tensor arguments | `_host_tensor('name')` | Function args: `%name: tensor<...>` | Preserves original parameter names |

## Tensor Loads & Stores

| Helion op | FX Target | MLIR Op | FX → MLIR Mapping |
| --------- | --------- | ------- | ----------------- |
| `x[tile_m, tile_k]` | `memory_ops.load` | `helion.load_tile_dynamic` | `load.args[0]` → source tensor SSA |
| `out[tile_m, tile_n] = val` | `memory_ops.store` | `helion.store_tile_dynamic` | `store.args[0]` → target tensor SSA |
| `hl.zeros([...])` | `full` | `helion.zero_tile` | Creates accumulator tile |
| `torch.addmm(acc, lhs, rhs)` | `aten.addmm.default` | `helion.call_torch` | `fn_name = "aten.addmm"` |

### Load Node Structure

Each `load` FX node contains:
```
load.args[0] = source tensor node (e.g., "x" from _host_tensor('x'))
load.args[1] = tile indices as string (e.g., "[tile_m, tile_k]")
load.name    = unique name (e.g., "load", "load_1")
```

The MLIR emitter extracts the source tensor name to:
1. Look up the corresponding kernel argument SSA (`%x`, `%y`)
2. Infer tile dimensions based on tensor position (LHS: `[M,K]`, RHS: `[K,N]`)
3. Preserve the FX node name via `fx_node` attribute

Each operation retains the originating FX node name (`fx_node = "load"`, `"store"`, …) so downstream passes can reconcile the MLIR with the original Helion FX graph.

## Status & Next Steps

- ✔️ **Loop topology**: reconstructed from Helion's `DeviceIR` (supports multiple roots / nested `_for_loop` blocks).
- ✔️ **Dynamic bounds**: tile sizes are handled via `affine.min` (concrete) or `arith.minsi` (symbolic) for partial tile handling.
- ✔️ **Symbolic tile sizes**: supports both concrete (`static_shapes=True`) and symbolic (`static_shapes=False`) tile sizes.
- ✔️ **FX provenance**: node names are threaded through attributes for debugging and later lowering.
- ✔️ **Dialect integration**: the bespoke `helion.*` ops are defined in C++ and registered via `helion-opt`.
- ⏳ **Math lowering**: arithmetic ops (`aten.addmm`, reductions, softmax pieces) are still opaque.
- ⏳ **Type/memory modeling**: `tensor_meta` strings are advisory; the eventual dialect will formalize buffer semantics.

## Building `helion-opt`

The dialect and driver live under `mlir/`. Configure and build the project using
the pre-installed MLIR toolchain exposed at `/mnt/fast/llvm-mlir`:

```bash
cmake -S . -B build
cmake --build build --target helion-opt
```

This produces `build/mlir/helion-opt`, which registers the Helion dialect along
with the affine/arith/func/tensor dialects the prototype currently emits. The
Python helper (`src/helion_fx_mlir/helion_mlir.py`) defaults to that binary when
present, falling back to upstream `mlir-opt` if needed.

Use the helper scripts (`examples/matmul.py`, `examples/attn.py`) to
inspect the stage-0 MLIR. When iterating on the dialect, re-run `helion-opt` (or
`mlir-opt -allow-unregistered-dialect` if the dialect is unavailable) to confirm
the generated IR parses cleanly. This README will evolve as we lock in the final
lowering rules.
