Here’s a concrete, step-by-step roadmap you can follow to go from Helion’s `DeviceIR` to an MLIR module with:

* outermost `hl.tile()` loop → `affine.parallel`
* inner loops → `affine.for`
* all math still as “torch calls” (no real lowering yet)

I’ll assume you’re writing this in Python using the MLIR Python bindings, but the structure is the same for a C++ emitter.

---

## 0. Decide on your MLIR “surface IR”

Before writing code, decide:

1. **Module/function shape**

   * One **MLIR module** per Helion kernel.
   * One **`func.func`** per `HostFunction` / `BoundKernel`.
   * Signature: use `tensor<...>` types (or `memref<...>` if you want to be closer to affine/memref world).
   * Example:

     ```mlir
     module {
       func.func @kernel(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
         ...
       }
     }
     ```

2. **Where to put “torch ops”**
   Since you don’t want to lower computations yet:

   * Either:

     * Use the **Torch dialect** (`torch.operator`, `torch.aten.*`) as stubs, or
     * Define a simple custom op like `helion.call_torch`:

       ```mlir
       %y = "helion.call_torch"(%x0, %x1)
            { fn_name = "aten.add" }
            : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
       ```
   * Plan for this now so your lowering of FX nodes has a clear target.

---

## 1. Hook into Helion objects

You want a function at the Python level roughly like:

```python
def lower_bound_kernel_to_mlir(bound_kernel) -> mlir.ir.Module:
    host_fn = bound_kernel.host_function       # or however it’s exposed
    device_ir = host_fn.device_ir              # DeviceIR instance
    ext_ast = host_fn.extended_ast             # annotated Python AST
    # ...
```

Define a **LoweringContext**:

```python
@dataclass
class LoweringContext:
    module: mlir.ir.Module
    func: mlir.ir.Operation        # func.func op
    body_builder: mlir.ir.OpBuilder
    host_fn: HostFunction
    device_ir: DeviceIR
    graph_infos: list[GraphInfo]
    fx_value_map: dict[torch.fx.Node, mlir.ir.Value]
    symint_map: dict[SymInt, mlir.ir.Value]    # for tile indices etc.
```

You’ll reuse this context everywhere.

---

## 2. Reconstruct loop structure (outermost `hl.tile` vs inner loops)

Loop structure is not fully explicit in Device IR — it comes from the extended AST / host IR — so you should:

1. **Walk `ext_ast` to build a loop tree**

   Conceptually:

   ```python
   class LoopNode:
       kind: Literal["tile", "for", "other"]
       ast_node: ast.AST                 # the actual For node / hl.tile call
       children: list["LoopNode"]        # nested loops / blocks
       basic_block_id: Optional[int]     # index into device_ir.graphs
   ```

   Algorithm:

   * Walk the extended AST with a custom visitor.
   * For each `for`:

     * Inspect the iterator:

       * `for tile in hl.tile(...):` → `kind = "tile"`.
       * Any other `for` (including loops inside tile) → `kind = "for"`.
   * Attach to each loop body the **basic block FX graph** id(s) that Helion associates with that segment of device code (this should be derivable from `HostFunction` metadata: the same information `generate_ast` uses to find which `GraphInfo` to use for a given AST block).

   Result: a **tree** starting from the kernel body, with a clear notion of which loops are tile loops and how they nest.

2. **Mark “outermost tile loops”**

   In that tree:

   * For each path from root → leaf, the **first `kind == "tile"`** you see is an “outermost tile loop”.
   * Any `tile` below that is “inner tile” (you can still lower them to `affine.for` if they are meant to be per-block loops).

   You can stash this as a boolean on `LoopNode`:

   ```python
   loop_node.is_outermost_tile: bool
   ```

---

## 3. Map tiling and shapes to MLIR indices

You now need to know **loop bounds** (for both `affine.parallel` and inner `affine.for`):

1. **Get extents from Helion’s meta/shape info**

   * Use the same shape information Helion already computed (fake/meta tensors, SymInts).
   * For each `hl.tile(...)`:

     * Extract:

       * total size (`N`)
       * tile size (`B`) or grid size (`num_tiles`) – whichever Helion exposes.
   * You want MLIR values for these: `N`, `B`, `num_tiles`. They can come from:

     * function arguments (`x.shape[0]` lowered to MLIR as some `dim` op), or
     * constants, or
     * SymInts you materialize as `index` values.

2. **Decide the parallel iteration space**

   For each **outermost tile loop**:

   * 1D: `for tile in hl.tile(N, block_size=B)` →

     * `affine.parallel (%t) = (0) to (num_tiles) step (1) { ... }`
   * 2D tiling (e.g. `hl.tile(x.size(0), x.size(1))`) →

     * `affine.parallel (%ty, %tx) = (0, 0) to (Ny_tiles, Nx_tiles) step (1, 1) { ... }`

   You don’t have to handle imperfect tiles perfectly right now; you can rely on Helion’s body code to do bounds checks, and just model the parallel **grid space**.

---

## 4. Build the MLIR skeleton

Implement a function:

```python
def create_mlir_module_for_kernel(host_fn) -> LoweringContext:
    ctx = mlir.ir.Context()
    ctx.allow_unregistered_dialects = True
    module = mlir.ir.Module.create()
    with mlir.ir.InsertionPoint(module.body):
        func_type = ...
        func_op = func_dialect.FuncOp(
            "kernel",
            func_type,
        )
    # Create an entry block, set up builder
    entry_block = func_op.add_entry_block()
    builder = mlir.ir.OpBuilder(entry_block)
    return LoweringContext(
        module=module,
        func=func_op,
        body_builder=builder,
        host_fn=host_fn,
        device_ir=host_fn.device_ir,
        graph_infos=host_fn.device_ir.graphs,
        fx_value_map={},
        symint_map={},
    )
```

Populate `func_type` from kernel arguments:

* For each argument Helion sees (meta tensors & scalars), pick:

  * `tensor<?x...xf32>` / `tensor<*xf32>` as a first cut.

---

## 5. Emit the loop nest

Write a recursive lowering function over your `LoopNode` tree:

```python
def lower_loop_tree(ctx: LoweringContext, node: LoopNode, builder: mlir.ir.OpBuilder):
    if node.kind == "tile" and node.is_outermost_tile:
        emit_outer_parallel_loop(ctx, node, builder)
    elif node.kind in ("tile", "for"):
        emit_sequential_loop(ctx, node, builder)
    else:
        emit_basic_block_or_stmt_list(ctx, node, builder)
```

### 5.1 `emit_outer_parallel_loop`

```python
def emit_outer_parallel_loop(ctx, loop_node, builder):
    # Compute (lower_bounds, upper_bounds, steps) from tile metadata.
    lbs, ubs, steps = compute_tile_bounds(ctx, loop_node)

    with builder:
        par_op = affine_d.AffineParallelOp(
            lbs, ubs, steps,
            # no reductions yet
        )
    # The region of AffineParallelOp has its own block; get a builder for it
    body_block = par_op.body.blocks[0]
    inner_builder = mlir.ir.OpBuilder(body_block)

    # Map the induction variables to SymInts / tile objects
    for iv, ast_tile_sym in zip(body_block.arguments, loop_node.tile_syms):
        ctx.symint_map[ast_tile_sym] = iv

    # Recurse into the loop body (children in the loop tree)
    for child in loop_node.children:
        lower_loop_tree(ctx, child, inner_builder)
```

`compute_tile_bounds` is the place where you read Helion’s tiling metadata (the same info `tile_ops` uses to know block/grid sizes) and return integer/`index` values for `affine.parallel`.

### 5.2 `emit_sequential_loop`

For inner loops (plain `for` and inner `hl.tile` treated as sequential):

```python
def emit_sequential_loop(ctx, loop_node, builder):
    lb, ub, step = compute_loop_bounds(ctx, loop_node)

    with builder:
        for_op = affine_d.AffineForOp(lb, ub, step)
    body_block = for_op.body.blocks[0]
    inner_builder = mlir.ir.OpBuilder(body_block)

    iv = body_block.arguments[0]
    ctx.symint_map[loop_node.induction_sym] = iv

    for child in loop_node.children:
        lower_loop_tree(ctx, child, inner_builder)
```

`compute_loop_bounds` is analogous but for regular (non-tile) loops.

### 5.3 Non-loop leaves: basic blocks

Leaves of the loop tree correspond to **basic blocks** (Device IR graphs) or simple statement lists. For those, call:

```python
def emit_basic_block_or_stmt_list(ctx, node, builder):
    # node.basic_block_id should be set where this AST region
    # is associated with a DeviceIR GraphInfo.
    if node.basic_block_id is not None:
        graph_info = ctx.graph_infos[node.basic_block_id]
        lower_fx_graph(ctx, graph_info.graph, builder)
    else:
        # No device code; maybe pure host control? Likely rare for device IR,
        # but you can ignore or add stubs.
        pass
```

---

## 6. Lowering a Torch FX graph to placeholder MLIR ops

Implement:

```python
def lower_fx_graph(ctx: LoweringContext, fx_graph: torch.fx.Graph, builder: mlir.ir.OpBuilder):
    # 1. Map graph inputs to MLIR values (kernel args, IVs, etc.)
    map_fx_placeholders(ctx, fx_graph, builder)

    # 2. Walk nodes in topological order
    for node in fx_graph.nodes:
        if node.op == "placeholder":
            continue
        elif node.op == "call_function":
            lower_call_function_node(ctx, node, builder)
        elif node.op == "call_method":
            lower_call_method_node(ctx, node, builder)
        elif node.op == "output":
            handle_output_node(ctx, node, builder)
        else:
            raise NotImplementedError(node.op)
```

### 6.1 Mapping placeholders

You need a consistent mapping from FX placeholders to MLIR values. Typically:

* For main graphs:

  * First few placeholders → kernel arguments (`%arg0`, `%arg1`, …).
  * Additional placeholders → loop IVs, constants, etc., which you can map via metadata Helion keeps (e.g. which SymInt each placeholder corresponds to).

Pseudo-code:

```python
def map_fx_placeholders(ctx, fx_graph, builder):
    for i, node in enumerate(fx_graph.nodes):
        if node.op != "placeholder":
            break
        mlir_val = resolve_placeholder_to_mlir_value(ctx, node)
        ctx.fx_value_map[node] = mlir_val
```

`resolve_placeholder_to_mlir_value` can consult:

* `ctx.func.entry_block.arguments`
* `ctx.symint_map`
* constants you materialize.

### 6.2 Lowering `call_function` nodes as torch calls

Since you want to “keep them as torch functions”, do something like:

```python
def lower_call_function_node(ctx, node, builder):
    target = node.target  # e.g., torch.add, operator.mul, etc.

    operand_vals = [ctx.fx_value_map[arg] for arg in node.args]
    # TODO: handle kwargs if needed

    fn_name_str = get_torch_symbol_name(target)  # e.g. "aten.add"

    # Emit a placeholder op
    with builder:
        result_types = [infer_mlir_type_from_fx_node(ctx, node)]
        op = builder.create(
            "helion.call_torch",
            operands=operand_vals,
            result_types=result_types,
            attributes={"fn_name": mlir.ir.StringAttr.get(fn_name_str)},
        )
    # FX nodes usually have 1 result for now; adjust if tuple
    ctx.fx_value_map[node] = op.result
```

You don’t need real type inference immediately; you can return some generic tensor type and refine later.

### 6.3 Outputs

For now, you can:

* If the graph returns a single tensor:

  * Call `func.return` with the MLIR value bound to the FX `output` node.

```python
def handle_output_node(ctx, node, builder):
    values = node.args[0]  # FX typically packs outputs as a tuple
    mlir_vals = [ctx.fx_value_map[v] for v in values]
    builder.create("func.return", operands=mlir_vals)
```

For helper graphs (like reductions’ combine functions) you may not even lower them yet — you can skip or stub them out unless you want a complete MLIR model of helpers too.

---

## 7. Tie-in with helper graphs (optional for v1)

For your first milestone, you can **ignore `HelperFunctionGraphInfo`** and treat calls like `hl.reduce` as opaque:

* In FX, the node corresponding to `_reduce(...)` will show up as a `call_function` with arguments including `combine_graph_id`.
* You can lower `_reduce` to a single `helion.call_torch` or `helion.reduce_stub` op and **store** the `combine_graph_id` as an attribute, without lowering the helper FX graph yet.

Later, when you want a more faithful MLIR, you can:

1. Lower helper graphs into separate `func.func @helper_...` functions.
2. Lower the main `_reduce` node to an op that calls those helper funcs.

---

## 8. Testing and iteration plan

Implement and test in this order:

1. **Minimal pipeline without device IR**

   * Hard-code a simple loop nest:

     ```python
     for tile in hl.tile(...):
         for i in range(...):
             y[i] = torch.add(x[i], x[i])
     ```
   * Fake the AST & loop tree and generate:

     * a single `affine.parallel` with one inner `affine.for`, and one `helion.call_torch`.
   * Verify with `mlir-opt` that the IR parses.

2. **Wire in real HostFunction / DeviceIR**

   * Run a real Helion kernel under your script.
   * Get `BoundKernel`, `host_function`, `device_ir`.
   * Build the loop tree using the extended AST.
   * For a kernel with one `hl.tile` and no control flow, generate MLIR and check:

     * one `affine.parallel` at the top,
     * maybe a few inner `affine.for`,
     * placeholder ops for all FX nodes.

3. **Add support for:

   * multiple nested tile loops (2D grid → `affine.parallel` with multiple IVs),
   * basic scalar `for` loops inside the tile (→ `affine.for`),
   * simple branching (you can lower host-side conditionals to `scf.if` wrapping blocks where you call `lower_fx_graph`).

4. **Refine types & bounds**

   * Replace dummy tensor types with real shapes/dtypes from Helion meta tensors.
   * Improve `compute_tile_bounds` to deal with odd sizes / partial tiles.

5. **Optional: start modeling helper graphs**

   * Map `HelperFunctionGraphInfo` to `func.func` in MLIR.
   * Replace `_reduce` / `_associative_scan` placeholders with MLIR ops that call those helpers.

---

If you’d like, next step we can pick a concrete Helion kernel from the docs, sketch the exact loop tree and then write out the corresponding MLIR we expect — that makes implementing `emit_outer_parallel_loop` / `emit_sequential_loop` almost mechanical.
