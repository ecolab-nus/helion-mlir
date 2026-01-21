# Block Sizes in Helion

This document explains how Helion tracks and manages block sizes during kernel compilation. Understanding these concepts is essential for the MLIR lowering process.

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

## Implications for MLIR Lowering

When generating MLIR:

1. **Grid loops** (LoopSpecBlockSizeSource, reduction=False):
   - Become `affine.parallel` dimensions
   - Trip count = `ceildiv(size, var)`

2. **Reduction loops** (ReductionLoopBlockSizeSource, reduction=True):
   - Become `affine.for` or `linalg` reduction patterns
   - May be unrolled or use special reduction strategies

3. **Fixed sizes** (FixedBlockSizeSource):
   - Can be inlined as constants
   - Enable more optimization opportunities

For MLIR symbol emission:
```mlir
// Symbolic block sizes (autotuned) → loom.get_symbol
%block_size_0 = "loom.get_symbol"() {name = "block_size_0"} : () -> index
%block_size_1 = "loom.get_symbol"() {name = "block_size_1"} : () -> index

// Concrete block sizes (specialized via hl.specialize() or fixed) → arith.constant
%block_size_2 = arith.constant 128 : index
```

> **Note**: When a block has a concrete `size` (e.g., from `hl.specialize(head_dim)` or `block_size=64`), the MLIR generator emits `arith.constant` instead of `loom.get_symbol`. This enables more optimization opportunities in downstream passes.
