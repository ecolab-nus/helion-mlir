# Block Sizes in Helion

This document explains how Helion tracks and manages block sizes during kernel compilation. Understanding these concepts is essential for reasoning about tiling, loop bounds, and kernel structure.

## Overview

Helion uses `BlockSizeInfo` to track information about each tiling dimension. These are stored in `CompileEnvironment.block_sizes` as a list, indexed by `block_id`.

## BlockSizeInfo Structure

```python
@dataclasses.dataclass
class BlockSizeInfo:
    block_id: int                    # Unique identifier (index in block_sizes list)
    size: torch.SymInt | int | AutoSize | None  # The iteration space size (NOT the tile size)
    var: torch.SymInt                # Symbolic variable representing the tile size
    reduction: bool                  # True if this is a reduction dimension
    block_size_source: BlockSizeSource  # How to get the actual tile size from Config
    debug_names: set[str]            # Human-readable names (e.g., "tile_m", "tile_n")
```

## Field Explanations

### `size` - The Iteration Space Extent

The `size` field represents the **total iteration space** for this dimension, NOT the tile size.

| Value | Meaning | Example |
|-------|---------|---------|
| `int` | Concrete extent | `size=512` means iterating over 512 elements |
| `torch.SymInt` | Symbolic extent | Dynamic dimension from tensor shape |
| `AutoSize` | Deferred size | Used by `hl.register_block_size()` before size is known |
| `None` | Multiple sizes | Block used with different extents (enables masking) |

**Example**: For `hl.tile(512)`, the `size=512` but the actual tile size (e.g., 64) comes from `block_size_source`.

### Effect of `static_shapes` on `size`

The kernel decorator's `static_shapes` setting controls whether tensor dimensions become concrete:

```python
@helion.kernel(static_shapes=True)   # size will be int (e.g., 256, 512)
@helion.kernel(static_shapes=False)  # size will be SymInt (e.g., s30, s48)
```

| `static_shapes` | `size` type | Example |
|-----------------|-------------|---------|
| `True` | `int` | `Size=256, Size=512` |
| `False` | `torch.SymInt` | `Size=s30, Size=s48` |

**With `static_shapes=False`** (from your test output):
```
Block 0: Size=s30, Var=u4, Reduction=False, Source=LoopSpecBlockSizeSource()
Block 1: Size=s48, Var=u5, Reduction=False, Source=LoopSpecBlockSizeSource()
Block 2: Size=128, Var=128, Reduction=True, Source=ReductionLoopBlockSizeSource(reduction_loop=0)
Block 3: Size=s34, Var=u7, Reduction=False, Source=LoopSpecBlockSizeSource()
```

Note that:
- `s30, s48, s34` are **backed SymInts** (symbolic but with known concrete values in `shape_env.var_to_val`)
- `u4, u5, u7` are **unbacked SymInts** (autotuned tile sizes, resolved from `Config`)
- Block 2's `size=128` is concrete because `head_dim` was specialized via `hl.specialize()`

### `var` - The Tile Size Symbol

The `var` field is a **symbolic variable** (`torch.SymInt`) representing the tile size at compile time.

- Used in tensor shape expressions (e.g., `tensor<?x?xf32>` where `?` corresponds to `var`)
- Registered in `HostFunction.expr_to_origin` with `BlockSizeOrigin`
- The concrete value comes from `Config` at code generation time

**Key insight**: `size` is the loop bound; `var` is the step/tile size.

### Possible `SymInt` States for `size` and `var`

When `size` or `var` is a `torch.SymInt`, there are a few distinct cases worth
separating.

Project policy note:

- Helion itself may keep advisory values for symbolic integers in
  `env.shape_env.var_to_val`.
- In this project, we no longer use `size_hint()` as a generic fallback to
  force unresolved symbols into concrete integers.
- The intended rule is:
  use directly resolved information when it exists; otherwise preserve the
  symbolic value.

#### `size` when it is a `SymInt`

`BlockSizeInfo.size` is usually one of these:

1. **Backed `SymInt` from a real tensor shape**
   - Typical source: tensor dimensions when `static_shapes=False`
   - Example printout style: `s30`, `s48`
   - Meaning:
     this is a symbolic extent, but Helion/PyTorch already know its current
     concrete value in the active `ShapeEnv`
   - How to inspect it:
     - `sym = bs.size._sympy_()`
     - `sym in env.shape_env.var_to_val`
       tells you whether the shape env has a recorded value for this symbol
     - `env.shape_env.var_to_val[sym]`
       gives that value
   - What this usually means:
     for a normal shape-derived `size`, the `var_to_val` entry is the actual
     current extent for this trace

2. **Unbacked `SymInt` created by the compiler**
   - Typical source: compiler-created symbols or expressions that are not tied
     directly to an input tensor dimension
   - Example printout style: often symbols whose names start with `u`
   - Meaning:
     this value is still symbolic, even if Helion has stored some bookkeeping
     information for it
   - How to inspect it:
     - `sym = bs.size._sympy_()`
     - check whether the expression contains unbacked symbols
     - in practice, Helion's own helper treats names like `u*` as unbacked
   - Important distinction:
     `shape_env.var_to_val[sym]` may still exist here, but for unbacked symbols
     it should be treated as Helion bookkeeping, not as permission for this
     project to force the symbol to a concrete loop bound

3. **Block-associated symbolic size**
   - Less common for `size` than for `var`, but possible if the extent itself is
     another registered symbolic block quantity
   - How to inspect it:
     - `env.get_block_id(bs.size)` or `env.resolve_block_id(bs.size)`
     - if this returns a block ID, the symbol is tied to a registered Helion
       block/grid origin
     - `HostFunction.current().expr_to_origin.get(bs.size._sympy_())`
       gives the origin object if one is registered

#### `var`

`BlockSizeInfo.var` is different from `size`: it is the tile-size symbol itself.

In the current Helion implementation, `var` is fundamentally:

1. **A Helion-created block-size `SymInt`**
   - Created by `CompileEnvironment.create_block_var(...)`
   - This uses `shape_env.create_unbacked_symint(...)`
   - So `var` is a compiler-created symbolic tile-size variable, not a tensor
     shape dimension from user input

2. **An unbacked symbol with a stored value in `ShapeEnv`**
   - Helion immediately stores a value for it in:
     `env.shape_env.var_to_val[var._sympy_()]`
   - This value is initially something like `64`
   - For `hl.register_block_size()` and similar flows, that value may later be
     refreshed once the true extent becomes known
   - In this project, that does **not** mean we should call `size_hint()` and
     collapse the symbol to an integer unless the lowering rule explicitly says
     to do so

3. **A symbol with registered block origin**
   - Helion registers:
     `HostFunction.current().expr_to_origin[bs.symbol()] = SymbolOrigin(BlockSizeOrigin(block_id))`
   - This is what lets later passes recognize that this `SymInt` is the tile
     size for block `bs.block_id`

4. **Sometimes reused across related blocks**
   - In reduction allocation, Helion may reuse an existing block var rather than
     creating a fresh one, so two `BlockSizeInfo` objects can intentionally
     share the same symbolic `var`

### How to Tell What Kind of `SymInt` You Have

Given:

```python
env = bound_kernel.env
hf = bound_kernel.host_function
bs = env.block_sizes[block_id]
```

use this checklist.

#### For `size`

```python
size = bs.size
```

1. Is it concrete already?

```python
isinstance(size, int)
```

2. Is it symbolic?

```python
isinstance(size, torch.SymInt)
```

3. What is the underlying symbolic expression?

```python
sym = size._sympy_()
```

4. Does `ShapeEnv` have a recorded value for this symbol?

```python
sym in env.shape_env.var_to_val
env.shape_env.var_to_val.get(sym)
```

5. Is this symbol tied to a registered block?

```python
env.get_block_id(size)
env.resolve_block_id(size)
hf.expr_to_origin.get(sym)
```

#### For `var`

```python
var = bs.var
sym = var._sympy_()
```

Useful queries:

```python
bs.symbol()                       # underlying sympy symbol
env.get_block_id(var)             # should point back to this block
hf.expr_to_origin.get(sym)        # usually BlockSizeOrigin(block_id)
env.shape_env.var_to_val.get(sym) # ShapeEnv's recorded value for this tile-size symbol
bs.from_config(config)            # actual configured tile size for a specific Config
bs.from_config_assert(config)     # same, but asserts non-None
```

### Practical Interpretation of These Tables

When reading Helion state, it helps to distinguish three questions:

1. **What symbolic object is this?**
   - use `._sympy_()`
   - use `hf.expr_to_origin.get(sym)`

2. **Is it tied to a block dimension?**
   - use `env.get_block_id(...)` / `env.resolve_block_id(...)`

3. **Is it already explicitly resolved, or is it still symbolic?**
   - use `sym in env.shape_env.var_to_val`
   - inspect `env.shape_env.var_to_val.get(sym)`
   - interpret that together with origin metadata, not as a blanket license to
     concretize the symbol

This distinction is important because:

- for a backed shape-derived `size`, `var_to_val` is usually the real concrete
  size for the current trace
- for `var`, and for other unbacked compiler-created symbols, `var_to_val` is
  often just Helion bookkeeping, not a statement that the symbol should be
  forced to an integer in this project

### What This Project Should Use to Resolve a `SymInt`

If you want to know how this project should treat a symbolic `size` or `var`,
use the following order of inspection:

1. Get the symbolic expression:

```python
sym = value._sympy_()
```

2. Check whether Helion attached an origin:

```python
origin_info = hf.expr_to_origin.get(sym)
```

3. If you need to know whether it is a registered block symbol, ask:

```python
env.get_block_id(value)
env.resolve_block_id(value)
```

4. If you need to know whether it is already explicitly resolved by the active
   shape environment, ask:

```python
env.shape_env.var_to_val.get(sym)
```

5. Only use config-based resolution when you are intentionally asking for the
   chosen tile size of `var`:

```python
bs.from_config(config)
```

What this project should **not** do anymore:

- call `env.size_hint(...)` as a generic fallback for unresolved symbols
- call `bs.size_hint()` as a generic fallback for unresolved block extents
- treat the absence of a direct symbolic resolution as a reason to invent a
  concrete bound from a hint

### `reduction` - Reduction Flag

| Value | Meaning |
|-------|---------|
| `False` | Grid/parallel loop dimension (from `hl.tile()`) |
| `True` | Reduction dimension (from `torch.sum`, `torch.amax`, etc.) |

Reduction dimensions are handled differently:
- They become inner loops in Triton (not grid dimensions)
- Their tile sizes come from `Config.reduction_loops` instead of `Config.block_sizes`

---

## BlockSizeSource Classes

The `block_size_source` field determines how the actual tile size is obtained from the `Config` during code generation.

### 1. LoopSpecBlockSizeSource (Default for `hl.tile()`)

```python
@dataclasses.dataclass
class LoopSpecBlockSizeSource(BlockSizeSource):
    def from_config(self, config: Config, block_size_info: BlockSizeInfo) -> int:
        # Get tile size from config.block_sizes array
        index = env.config_spec.block_sizes.block_id_to_index(block_size_info.block_id)
        return config.block_sizes[index]
```

**When used**: Default for `hl.tile()` without explicit `block_size` argument.

**Effect**: The tile size is **autotuned** and read from `Config.block_sizes`.

**Example**:
```python
for tile_m, tile_n in hl.tile([M, N]):
    # tile_m -> BlockSizeInfo(block_id=0, source=LoopSpecBlockSizeSource())
    # tile_n -> BlockSizeInfo(block_id=1, source=LoopSpecBlockSizeSource())
```

### 2. FixedBlockSizeSource (Explicit `block_size=`)

```python
@dataclasses.dataclass
class FixedBlockSizeSource(BlockSizeSource):
    value: int | torch.SymInt

    def from_config(self, config: Config, block_size_info: BlockSizeInfo) -> int | torch.SymInt:
        return self.value  # Always returns the fixed value
```

**When used**: When user explicitly specifies `block_size` in `hl.tile()`.

**Effect**: The tile size is a **compile-time constant**, not autotuned.

**Example**:
```python
for tile in hl.tile(M, block_size=64):
    # -> BlockSizeInfo(block_id=0, source=FixedBlockSizeSource(value=64))
```

### 3. ReductionLoopBlockSizeSource (Reduction Operations)

```python
@dataclasses.dataclass
class ReductionLoopBlockSizeSource(BlockSizeSource):
    reduction_loop: int  # Index in config.reduction_loops

    def from_config(self, config: Config, block_size_info: BlockSizeInfo) -> int | None:
        if len(config.reduction_loops) <= self.reduction_loop:
            # Fallback: use next power of 2 of the extent
            return max(1, next_power_of_2(block_size_info.size_hint()))
        return config.reduction_loops[self.reduction_loop]
```

**When used**: Automatically created by reduction operations like `torch.sum`, `torch.amax`, `torch.mean`.

**Effect**: The reduction tile size comes from `Config.reduction_loops` or defaults to a power of 2.

This snippet describes upstream Helion behavior. In `helion-mlir`, we no longer
use `size_hint()` as a generic fallback when lowering unresolved symbolic
extents.

**Example**:
```python
# In attention kernel:
torch.amax(qk, -1)  # qk shape: [B, M, head_dim]
# -> BlockSizeInfo(block_id=2, size=128, reduction=True, 
#                  source=ReductionLoopBlockSizeSource(reduction_loop=0))
```

---

## Allocation Flow

### `hl.tile()` Allocation

```
hl.tile([M, N])
    │
    ├─► allocate_block_size(size=M, source=LoopSpecBlockSizeSource())
    │       → block_id=0, var=u0
    │
    └─► allocate_block_size(size=N, source=LoopSpecBlockSizeSource())
            → block_id=1, var=u1
```

### Reduction Allocation

```
torch.sum(tensor, dim=-1)
    │
    └─► allocate_reduction_dimension(size=last_dim_size)
            │
            └─► allocate_block_size(size=..., reduction=True,
                                    source=ReductionLoopBlockSizeSource(0))
                    → block_id=2, var=u2
```

---

## Example: Attention Kernel

Given this kernel:
```python
for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):  # Block 0, 1
    ...
    for tile_n in hl.tile(v_view.size(1)):               # Block 3
        torch.amax(qk, -1)  # Reduction over head_dim     # Block 2
        torch.sum(p, -1)    # Reuses Block 2
```

The resulting `block_sizes`:

| block_id | size | var | reduction | source | debug_names |
|----------|------|-----|-----------|--------|-------------|
| 0 | 256 | u0 | False | LoopSpecBlockSizeSource() | {'tile_b'} |
| 1 | 512 | u1 | False | LoopSpecBlockSizeSource() | {'tile_m'} |
| 2 | 128 | 128 | True | ReductionLoopBlockSizeSource(0) | {} |
| 3 | 512 | u3 | False | LoopSpecBlockSizeSource() | {'tile_n'} |

Note: Block 2 has `var=128` (concrete) because the reduction dimension was specialized via `hl.specialize(head_dim)`.

---

## Loop Bounds and Block Sizes

The main reason block sizes matter is that they define how Helion turns an
iteration space into tile loops.

At a high level:

- `size` tells you the total extent of a dimension.
- `var` tells you the tile size used to walk that extent.
- The number of loop iterations is therefore:
  `ceildiv(loop_upper_bound, tile_size)`.

For simple outer tiled loops, the upper bound is usually just the total extent.
For inner block loops, the upper bound can itself be an expression involving
other tile properties and block sizes.

### Where Loop-Bound Information Comes From

For the current implementation, loop-bound reasoning uses the following pieces
of information:

1. The `_for_loop` node in Helion Device IR.
   - lower bounds are stored in `node.args[1]`
   - upper bounds are stored in `node.args[2]`
   - an optional explicit step may appear in `node.args[4]`
2. `ForLoopGraphInfo.block_ids`
   - identifies which `BlockSizeInfo` controls that loop
3. `BlockSizeInfo`
   - provides the total extent (`size`)
   - provides the symbolic tile-size variable (`var`)
   - tells whether the block is a reduction dimension
4. Origin metadata plus `shape_env`
   - resolves symbolic expressions back to specific block IDs or concrete values
5. Precomputed fallback trip-count metadata
   - used when the loop bounds cannot be reconstructed directly from the FX node

More concretely, the current lowering uses these sources in this order:

1. The FX `_for_loop` node itself.
   - This is the primary source for loop bounds.
   - The lower bound comes from `node.args[1]`.
   - The upper bound comes from `node.args[2]`.
   - If present, `node.args[4]` is interpreted as the loop step.
2. The loop-to-block mapping from `ForLoopGraphInfo.block_ids`.
   - This tells us which `BlockSizeInfo` supplies the tile size for that loop.
3. The block-size metadata for that block.
   - The tile size comes from the block's `var`/origin information.
   - The total extent comes from `size`, or from directly resolved symbolic or
     concrete information Helion can recover without forcing a hint.
4. Origin metadata and the shape environment.
   - Used to resolve symbolic expressions such as block-size symbols and known tensor dimensions.
5. Fallback trip-count metadata.
   - Used only when the FX-bound path cannot reconstruct the trip count directly.

### Simple Bound Case

For a standard tiled loop such as:

```python
for tile_m, tile_n in hl.tile([M, N]):
    ...
```

the conceptual bounds are:

- outer extent for `tile_m`: `M`
- tile size for `tile_m`: `block_m`
- trip count for `tile_m`: `ceildiv(M, block_m)`

and similarly for `tile_n`.

This is why the key distinction in `BlockSizeInfo` is:

- `size` = total iteration-space extent
- `var` = tile size used to step through that extent

### Expression-Based Bound Case

The more interesting case is when the loop upper bound is not a raw tensor
extent, but an expression built from other tile properties.

Example from `mamba_chunk_scan`:

```python
for tile_k in hl.tile((tile_m.id + 1) * block_m, block_size=block_k):
    ...
```

Here the upper bound for the `tile_k` loop is:

```text
(tile_m.id + 1) * block_m
```

This means the bound depends on:

- `tile_m.id`: which tile of `m` we are currently processing
- `block_m`: the tile size associated with the `m` dimension
- `block_k`: the tile size used to iterate the inner `k` loop

So the trip count for `tile_k` is conceptually:

```text
ceildiv((tile_m.id + 1) * block_m, block_k)
```

This is the key connection between block sizes and loop bounds:

- block sizes are not only used as tile extents
- they can also appear inside the expressions that define other loop bounds

### How the Current Loop-Bound Calculation Works

Given a `_for_loop` node, the current logic computes the loop trip count as
follows:

1. Identify which logical block controls the loop.
   - Read `ForLoopGraphInfo.block_ids`.
   - Canonicalize the chosen block ID through the alias map when needed.
2. Try to derive the bounds directly from the FX `_for_loop` node.
   - Read the lower-bound list from `node.args[1]`.
   - Read the upper-bound list from `node.args[2]`.
   - Read the optional step from `node.args[4]` if present.
3. Validate the currently supported shape of the loop.
   - Lower bound must be `0`.
   - Step must be `1`.
   - The current implementation assumes a single loop dimension for this path.
4. Lower the upper-bound expression to a symbolic/concrete scalar expression.
   - Literals stay literals.
   - Symbolic dimensions are resolved through `shape_env` and origin metadata.
   - Arithmetic such as `add`, `sub`, `mul`, and `floordiv` is preserved structurally.
5. Resolve the tile size for the loop's block.
   - If the block is effectively static, use the concrete tile size.
   - Otherwise use the symbolic tile-size variable associated with that block.
6. Compute the trip count:

```text
trip_count = ceildiv(upper_bound, tile_size)
```

7. If the FX-bound path is unavailable, use the precomputed fallback trip count.

This is the key algorithmic point:

- loop bounds are computed from the loop's upper-bound expression
- loop iteration counts are then derived by dividing by the block's tile size

### What Counts as the Upper-Bound Expression

The upper-bound expression can be:

- a plain extent such as `M`
- a symbolic tensor dimension
- a tile-size symbol
- an arithmetic expression built from any of the above

Examples:

```text
M
head_dim
tile_m.id + 1
(tile_m.id + 1) * block_m
```

The final trip count is always derived from that upper-bound expression and the
tile size for the current block:

```text
ceildiv(upper_bound_expr, current_block_tile_size)
```

### Current Loop-Bound Assumptions

The current loop-bound support assumes:

- lower bound is `0`
- step is `1`
- the FX-bound-first path is for a single logical loop dimension

So the supported model is:

```text
for i in range(0, upper_bound, 1)
```

where `upper_bound` may be:

- a plain extent such as `M`
- a symbolic size from tensor metadata
- an expression involving tile properties and block sizes

If any of these assumptions are violated, the current direct loop-bound path is
not intended to handle the case. In particular:

- non-zero lower bounds are outside the supported model
- non-unit steps are outside the supported model
- these cases require either rejection or some different lowering strategy

### Tile Properties as Bound Helpers

Tile properties are another view of the same block-size information.

Conceptually:

- `tile.id`
  which tile number we are on along that dimension
- `tile.begin`
  the first element index covered by the current tile
- `tile.end`
  the one-past-end element index for the current tile
- `tile.index`
  the contiguous element indices inside the tile

These matter for loop bounds because expression-based bounds often use them.

Typical examples:

- `tile.begin` is used to build offsets into tensors
- `tile.id` is used to make later loop bounds depend on the current outer tile
- `tile.index + base` describes a contiguous range whose extent is the tile size

This is why an expression like:

```python
(tile_m.id + 1) * block_m
```

is naturally a loop bound:

- `tile_m.id` tells you how many `m`-tiles have been covered so far
- `block_m` converts that tile count back into an element-space bound
- adding `1` means "include the current tile"

### Why `BlockSizeInfo` Fields Matter for Bounds

From the perspective of loop bounds:

- `block_id`
  ties a loop or tile property back to the right logical tiling dimension
- `size`
  gives the overall extent being covered
- `var`
  gives the tile-size symbol used in `ceildiv(..., tile_size)`
- `reduction`
  distinguishes outer grid-style tiling from inner reduction/block loops
- `debug_names`
  make it possible to recognize when two block IDs from different grid groups
  refer to the same logical dimension

`block_size_source` is still important upstream in Helion, but conceptually it
only explains where the chosen tile size came from.  Once `BlockSizeInfo` has
been built, loop-bound reasoning mainly depends on `size`, `var`, `reduction`,
and the origin metadata attached to symbolic expressions.

---

## Multi-Grid Kernels and Block Size Aliasing

When a kernel uses `hl.barrier()` to separate two stages, Helion creates
**multiple grid groups** — each with its own set of block IDs.  The same
logical dimension (e.g. *M*) appears as a different `block_id` in each grid,
even though both blocks tile the same source dimension.

### How Grid Groups Are Formed

```python
# Stage 1 — grid group 0
for tile_m, tile_n, tile_k in hl.tile([m, n, k]):   # block_ids [0, 1, 2]
    ...

hl.barrier()

# Stage 2 — grid group 1
for tile_m, tile_n in hl.tile([m, n]):               # block_ids [3, 4]
    ...
```

Helion exposes these groups via `device_ir.grid_block_ids`:

```
grid_block_ids: [[0, 1, 2], [3, 4]]
root_ids:       [0, 1]
```

### Identifying the Source of a Block Size

Each `BlockSizeInfo` carries a `debug_names` set that records which user-level
tile variable the block originated from:

| block_id | debug_names   | numel | grid group |
|----------|---------------|-------|------------|
| 0        | `{'tile_m'}`  | s97   | 0          |
| 1        | `{'tile_n'}`  | s20   | 0          |
| 2        | `{'tile_k'}`  | s98   | 0          |
| 3        | `{'tile_m'}`  | s97   | 1          |
| 4        | `{'tile_n'}`  | s20   | 1          |

Blocks **0 and 3** both have `debug_names={'tile_m'}` and the same `numel`.
Blocks **1 and 4** both have `debug_names={'tile_n'}` and the same `numel`.
This tells us they tile the same logical dimension and should share the same
canonical identity.

### Building the Alias Map

`LoweringContext._build_block_id_alias_map()` walks the grid groups in order
and builds a canonical mapping.  Two blocks are aliased when they share the
same `(frozenset(debug_names), numel)` key:

```python
# Pseudocode
canonical = {}   # (frozenset(debug_names), numel) → first block_id
alias    = {}    # every block_id → its canonical block_id

for grid_group in grid_block_ids:
    for bid in grid_group:
        info = block_sizes[bid]
        key  = (frozenset(info.debug_names), info.numel)
        if key not in canonical:
            canonical[key] = bid      # first occurrence is canonical
        alias[bid] = canonical[key]
```

Result for the split-K example:

```
alias = {0: 0, 1: 1, 2: 2, 3: 0, 4: 1}
        ──────────────  ──────────────
        grid 0 (self)   grid 1 → canonical
```

### Why Aliasing Matters

Aliasing matters because loop-bound expressions should keep the same logical
meaning across grid groups.

If stage 0 and stage 1 both contain a `tile_m` dimension, then expressions
built from `tile_m.id`, `tile_m.begin`, or the corresponding tile size should
refer to the same logical quantity, even if Helion assigned different raw
`block_id` values in each stage.

That is why the alias map is based on:

- `debug_names`
- `numel`

and why later reasoning uses the canonical block identity rather than the raw
per-grid block ID.
