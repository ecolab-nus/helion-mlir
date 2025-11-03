* **Part A (Python):** Helion device-IR ➜ normalized **JSON**
* **Part B (C++/MLIR):** JSON ➜ **MLIR text** (keep PyTorch math in the `torch` dialect; convert `hl.tile` to `affine`/`scf`, preferring `affine`)

I’ve included: repo layout, exact JSON schema, builder algorithms (with affine/scf decision rules), edge-tile handling, runtime asserts, example IO, pass pipelines, tests, and CLI specs—with references to the relevant MLIR pieces.

---

# 0) Goals & non-goals

**Goals**

* Preserve PyTorch math semantics as high-level ops (don’t lower math to `linalg`).
* Lower `hl.tile` into **structured loop nests** using **`affine.for`** whenever legality allows (fallback to **`scf.for`**), with **edge-tile clamping via `affine.min`**; slice tiles using **`memref.subview`** (buffer form) or **`tensor.extract_slice`** (tensor form). ([mlir.llvm.org][1])
* Emit readable **textual MLIR** modules and be compatible with standard MLIR toolchains (`mlir-opt`, `mlir-translate`, etc.). ([mlir.llvm.org][2])

**Non-goals**

* No lowering to `linalg`/`vector` for math.
* No device codegen yet (e.g., `gpu`/`llvm` backends) beyond representing memory with `memref`.

---

# 1) Repos & build system

```
helion-mlir/
  helion2json/                # Python package
    __init__.py
    cli.py
    core/
      __init__.py
      capture.py
      cli.py
    tests/
      test_matmul.py
  json2mlir/                  # C++ project
    include/helion_mlir/
      JsonLoader.h
      Builder.h
      TypeMapper.h
      TorchBridge.h
      TileLowering.h
      Diagnostics.h
    lib/
      JsonLoader.cpp
      Builder.cpp
      TypeMapper.cpp
      TorchBridge.cpp
      TileLowering.cpp
      main.cpp              # CLI: json2mlir
    cmake/
    CMakeLists.txt
    tests/
      FileCheck/
        matmul_tiled.json
        matmul_tiled.mlir.expected
        ...
```

**Dependencies**

* Helion→JSON exporter (Python): Python 3.10+, `pydantic` (schema validation), `orjson`, `click`.
* JSON→MLIR lowering (C++): LLVM/MLIR (current head or stable branch), `nlohmann/json` (or RapidJSON), `gtest`. If you keep PyTorch ops in MLIR, add **Torch-MLIR** as a dependency to register the `torch` dialect. ([GitHub][3])

---

# 2) JSON interchange format (authoritative)

Use a single **module** with **functions**, **values**, and **ops**. Keep it deliberately compact and fully typed.

```json
{
  "version": 1,
  "module": {
    "name": "helion_module",
    "funcs": [
      {
        "name": "main",
        "args": [
          {"id": "x", "type": {"tensor": {"shape": ["?", "?"], "elem": "f32"}}},
          {"id": "y", "type": {"tensor": {"shape": ["?", "?"], "elem": "f32"}}}
        ],
        "rets": [
          {"type": {"tensor": {"shape": ["?", "?"], "elem": "f32"}}}
        ],
        "body": [
          {"op": "hl.assert_eq", "lhs": "dim(x,1)", "rhs": "dim(y,0)", "msg": "size mismatch"},
          {"op": "hl.tile.begin",
           "iters": ["m","n"],
           "sizes": [64, 64],                 // static tile sizes preferred
           "result": ["tile_m", "tile_n"]     // loop IVs (offsets)
          },
          {"op": "hl.zeros", "shape": ["size(tile_m)","size(tile_n)"], "dtype": "f32", "result": "acc"},
          {"op": "hl.tile.begin",
           "iters": ["k"],
           "sizes": [64],
           "result": ["tile_k"]
          },
          {"op": "torch.addmm",
           "result": "acc2",
           "args": [
             "acc",
             {"slice": {"base":"x", "offsets":["tile_m","tile_k"], "sizes":["size(tile_m)","size(tile_k)"]}},
             {"slice": {"base":"y", "offsets":["tile_k","tile_n"], "sizes":["size(tile_k)","size(tile_n)"]}}
           ],
           "attrs": {"beta": 1.0, "alpha": 1.0}       // mirrors aten.addmm
          },
          {"op": "hl.tile.end"},                       // ends K-tile loop, reduces into acc
          {"op": "hl.call", "callee":"epilogue",
           "args": ["acc2","tile_m","tile_n"], "result":"tile_out"},
          {"op": "hl.store_slice", "dst":"out",
           "offsets":["tile_m","tile_n"], "src":"tile_out"},
          {"op": "hl.tile.end"},                       // ends MxN tile loop
          {"op": "hl.return", "values": ["out"]}
        ]
      }
    ]
  }
}
```

**Conventions**

* **Types**: `tensor` (preferred) or `memref`. Each has `shape` and `elem`. Use `"?"` for dynamic dims.
* **Indices**: symbolic `'m','n','k'` may reference `dim(tensor, axis)`; the loader materializes them via `memref.dim` / `tensor.dim`.
* **Slices**: `{"slice": {base, offsets[], sizes[]}}` expresses tile views; loader decides `extract_slice` vs `subview`. ([mlir.llvm.org][4])
* **`hl.tile.begin/end`** pairs define **perfect loop nests** over given iteration spaces with **`sizes` as steps** and **tile-local sizes computed with min() at edges**. (See §5)

Validate JSON with `pydantic` and reject ambiguous constructs (e.g., dangling `hl.tile.begin`).

---

# 3) Part A (Python) — Helion device-IR ➜ JSON

**Inputs**

* A Helion kernel in Python. Your existing Helion compile step yields a **device IR** object graph.

**Steps**

1. **Capture device IR**: expose a read-only Python API on Helion’s IR (nodes, edges, attributes, shapes, dtypes, `hl.tile`, indexing, calls).
2. **Normalize**:

   * Canonicalize tensor/memref types (rank, elem type).
   * Canonicalize `hl.tile` to the established form (`iters`, `sizes`).
   * Normalize slices/indexing (`base`, `offsets`, `sizes`).
   * Hoist static `tile_sizes` into `sizes` (ints); keep dynamic tile sizes as symbolic (strings) if required.
3. **Infer symbolic dims**: attach symbolic names (`m,n,k`) to any `dim(x,i)` the program uses.
4. **Emit JSON** per schema above (minify or pretty via CLI flag).
5. **Tooling**: `helion2json` CLI.

**CLI**

```
$ helion2json kernel.py --entry main --out dev_ir.json
# Options:
#   --tile-size-m 64 --tile-size-n 64 --tile-size-k 64   # default tiles if absent
#   --tensor|--memref                                   # select value model
#   --validate                                          # schema check
```

**Notes**

* If Helion already has a serialized device IR, you may bypass normalization and adapt a translator to the schema above.

---

# 4) Part B (C++) — JSON ➜ MLIR

## 4.1 Dialect & context setup

Register at least: `builtin`, `func`, `affine`, `scf`, `arith`, `memref`, `tensor`, `cf`, and **`torch`** (from Torch-MLIR) to keep PyTorch math ops as `torch.aten.*`. ([mlir.llvm.org][5])

```cpp
mlir::DialectRegistry registry;
registry.insert<mlir::func::FuncDialect,
                mlir::affine::AffineDialect,
                mlir::scf::SCFDialect,
                mlir::arith::ArithDialect,
                mlir::memref::MemRefDialect,
                mlir::tensor::TensorDialect,
                mlir::cf::ControlFlowDialect,
                mlir::torch::TorchDialect>(); // from torch-mlir
mlir::MLIRContext ctx(registry);
ctx.loadAllAvailableDialects();
```

## 4.2 High-level flow

```
main() -> parse JSON -> ModuleOp -> for each func:
  create func.func @name(args)->rets
  map JSON values -> SSA (ValueTable)
  lower body seq:
    - assertions -> cf.assert
    - tile begin/end -> loop nests (affine preferred)
    - slices -> subview/extract_slice
    - torch ops -> torch.aten.*
    - calls/returns -> func.call/func.return
print module to stdout
```

**Runtime asserts**: translate `hl.assert_eq` to `cf.assert` with `arith.cmpi`. This exactly matches the Python `assert` intent. ([mlir.llvm.org][6])

## 4.3 Type mapping

* `{"tensor": {"shape": [...], "elem": "f32"}}` ➜ `RankedTensorType`
* `{"memref": {"shape": [...], "elem": "f32"}}` ➜ `MemRefType`
* Unknown device memory can remain in default memory space for now (extend later with address spaces if needed). Use `memref` for in-place tiles; use `tensor` for functional style. **Both are supported by MLIR; bufferization bridges between them.** ([mlir.llvm.org][7])

## 4.4 Slices / tiles

* **Tensor form**: `tensor.extract_slice %t[offs][sizes][strides]`
* **Buffer form**: `memref.subview %m[offs][sizes][strides]`
  Edge-tile sizes are clamped: `sizes = affine.min(tileSize, bound - iv)` (see §5). ([mlir.llvm.org][4])

## 4.5 Torch ops

Map JSON ops named `torch.*` to **Torch-MLIR dialect** ops (`torch.aten.addmm`, etc.), keeping semantics intact (no lowering to `linalg`). Keep attributes (e.g., `alpha`, `beta`). This is the standard way to **preserve PyTorch math at MLIR level** today. ([GitHub][3])

> If a particular ATen op isn’t present, emit a `func.call` to a symbol `@aten.<name>` and mark it for later legalization (but try to rely on `torch` whenever possible).

---

# 5) Lowering `hl.tile` ➜ `affine` / `scf`

## 5.1 Decision rule

**Prefer `affine.for`** when all are true:

* Lower/upper bounds are affine expressions of existing loop IVs and **symbol** values (e.g., `memref.dim` results).
* **Step is a positive compile-time integer** (constant tile size).
* Indexing of any `memref` in the tile is affine in those IVs/symbols.
  If any of the above is false (e.g., dynamic step size), **fallback to `scf.for`**. ([mlir.llvm.org][1])

## 5.2 Generated shape (2D outer, 1D inner)

Given iteration extents `%M = dim(x,0)`, `%N = dim(y,1)`, `%K = dim(x,1)` and tile sizes `Tm,Tn,Tk`:

* **Outer nest** (M×N tiles):

  ```
  affine.for %im = 0 to %M step Tm {
    %ms = affine.min #map(%im, %M)  // min(Tm, %M - %im)
    affine.for %in = 0 to %N step Tn {
      %ns = affine.min #map(%in, %N) // min(Tn, %N - %in)
      ...
    }
  }
  #map = affine_map<(d0)[s0] -> (Tm, s0 - d0)>
  ```

  If `Tm`/`Tn` are dynamic ➜ use `scf.for` and compute `%ms/%ns` with `affine.min` or `arith.min` equivalents; the loop body is otherwise identical. ([mlir.llvm.org][1])

* **Inner K-tile loop**:

  ```
  affine.for %ik = 0 to %K step Tk {
    %ks = affine.min #map(%ik, %K)
    // tile slices for X[%im:%ms, %ik:%ks] and Y[%ik:%ks, %in:%ns]
  }
  ```

* **Tile materialization**:

  * tensor path: `tensor.extract_slice`
  * memref path: `memref.subview`
    Both are the standard ops to carve subregions for tile compute. ([mlir.llvm.org][4])

* **Parallel tiles** (optional): if you want implicit parallelism, outer nest can be `scf.forall` instead of nested for’s; later, transforms can turn it into `scf.parallel` or back to `scf.for`. Keep this as a future toggle. ([mlir.llvm.org][8])

---

# 6) Concrete MLIR emission (sketch)

Below shows **memref** form with `affine` (happy path). Use `tensor.*` variants similarly.

```mlir
module {
  func.func @main(%x: memref<?x?xf32>, %y: memref<?x?xf32>) -> memref<?x?xf32> {
    %c0   = arith.constant 0 : index
    %Tm   = arith.constant 64 : index
    %Tn   = arith.constant 64 : index
    %Tk   = arith.constant 64 : index

    %M = memref.dim %x, 0 : memref<?x?xf32>
    %K1 = memref.dim %x, 1 : memref<?x?xf32>
    %K2 = memref.dim %y, 0 : memref<?x?xf32>
    %N = memref.dim %y, 1 : memref<?x?xf32>

    %ok = arith.cmpi eq, %K1, %K2 : index
    cf.assert %ok, "size mismatch k != k2" : i1

    %out = memref.alloc(%M, %N) : memref<?x?xf32>

    affine.for %im = 0 to %M step 64 {
      %ms = affine.min affine_map<(d0)[s0] -> (64, s0 - d0)>(%im)[%M]
      affine.for %in = 0 to %N step 64 {
        %ns = affine.min affine_map<(d0)[s0] -> (64, s0 - d0)>(%in)[%N]

        // acc tile buffer (zeroed via a fill loop not shown)
        %acc = memref.alloc(%ms, %ns) : memref<?x?xf32>

        affine.for %ik = 0 to %K1 step 64 {
          %ks = affine.min affine_map<(d0)[s0] -> (64, s0 - d0)>(%ik)[%K1]

          %xT = memref.subview %x[%im, %ik] [%ms, %ks] [1,1]
                 : memref<?x?xf32> to memref<?x?xf32>
          %yT = memref.subview %y[%ik, %in] [%ks, %ns] [1,1]
                 : memref<?x?xf32> to memref<?x?xf32>

          // Keep math in torch dialect:
          // torch.aten.addmm(acc, xT, yT) -> acc'
          %acc2 = "torch.aten.addmm"(%acc, %xT, %yT)
                   {alpha = 1.0 : f64, beta = 1.0 : f64}
                   : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> memref<?x?xf32>
          // (If torch expects tensors, convert around with cast/alloc + copies.)
        }

        // epilogue(acc2, tile coords) — either a call or a torch op
        // Then write to out tile:
        %outT = memref.subview %out[%im, %in] [%ms, %ns] [1,1]
                : memref<?x?xf32> to memref<?x?xf32>
        // copy acc2 -> outT (elided)
      }
    }
    return %out : memref<?x?xf32>
  }
}
```

* `affine.min` and `affine.for` are the standard mechanisms to clamp edge tiles and express tiling. ([mlir.llvm.org][1])
* If legality fails (e.g., dynamic step), switch to `scf.for`—semantics are equivalent but without affine restrictions. ([mlir.llvm.org][8])

---

# 7) Implementation details (C++)

## 7.1 Reader & symbol table

* `JsonLoader` builds:

  * **ValueTable**: JSON `id` → `mlir::Value`
  * **DimTable**: `"dim(x,1)"` strings → computed `memref.dim`/`tensor.dim` Values
  * **TileStack**: push on `hl.tile.begin`, pop on `hl.tile.end` (supports nesting)

## 7.2 Builder API (selected)

```cpp
struct BuildCtx {
  mlir::MLIRContext &ctx;
  mlir::OpBuilder builder;
  mlir::Location loc;
  mlir::ModuleOp module;
  llvm::StringMap<mlir::Value> values; // SSA map
};

mlir::Value getDim(BuildCtx&, mlir::Value tensorOrMemref, int64_t axis); // emits memref.dim/tensor.dim
mlir::Operation* emitAssertEq(BuildCtx&, mlir::Value lhs, mlir::Value rhs, llvm::StringRef msg); // cf.assert

struct TileConfig {
  // wanted step sizes; nullopt = dynamic / fallback to scf
  std::optional<int64_t> step;
  mlir::Value ub;      // loop upper bound (e.g. dim)
};

struct TileNest {
  // returns (outer loop op, iv, innerGuard builder)
  mlir::Operation* makeLoopAffineOrScf(BuildCtx&, const TileConfig&, mlir::Value *iv, mlir::Value *clampedSize);
};
```

**Edge size**:
`%sz = affine.min (step, ub - iv)` in affine case; `scf` case compute `%rem = arith.subi ub, iv; %sz = arith.minsi step, %rem`. ([mlir.llvm.org][1])

## 7.3 Slices

* Tensor: `tensor::ExtractSliceOp`
* Memref: `memref::SubViewOp`
* Strides default to 1; use the clamped sizes from above. ([mlir.llvm.org][4])

## 7.4 Torch bridge

* Create ops in the `torch` dialect with the right operand kinds (tensor vs buffer). If your surrounding IR is memref-based, insert **materialization**:

  * `memref` → `tensor`: `bufferization.to_tensor`
  * back to `memref`: `bufferization.to_memref` (or explicit copy)
    Alternative: choose **tensor-centric** IR globally and bufferize later (recommended). ([mlir.llvm.org][7])

---

# 8) Pass pipelines (developer & CI)

Run a light canonical pipeline after emission:

* **Tensor path**
  `-canonicalize -cse`
  (optionally) `-one-shot-bufferize --bufferize-function-boundaries` later if you need memrefs. ([mlir.llvm.org][7])

* **Memref path**
  `-canonicalize -cse`
  (optional, if any affine accesses are present) `-affine-simplify-structures -affine-loop-fusion -affine-expand-index-ops` etc. ([mlir.llvm.org][9])

(Keep transforms minimal since you want to preserve PyTorch math.)

---

# 9) Testing strategy

**Unit tests (C++)**

* Type mapping (dynamic and static shapes).
* `hl.tile` lowering matrix (1D/2D/3D, static/dynamic steps).
* Slice formation & edge clamping.

**Golden/IR tests (FileCheck)**

* `json2mlir < tests/FileCheck/matmul_tiled.json | FileCheck tests/.../matmul_tiled.mlir.expected`

**End-to-end**

* `helion2json kernel.py | json2mlir | mlir-opt -verify-diagnostics`

**Assertions**

* Verify `k == k2` becomes `cf.assert` + message. ([mlir.llvm.org][6])

---

# 10) CLI contracts

**A) helion2json (Python)**

```
Usage: helion2json <kernel.py> --entry <name> [options]
Options:
  --tile-size-m INT --tile-size-n INT --tile-size-k INT
  --default-dtype f32|f16|bf16
  --tensor | --memref      # choose representation
  --out FILE | stdout
  --validate
```

**B) json2mlir (C++)**

```
Usage: json2mlir [--tensor|--memref] [--prefer-affine] [--prefer-scf]
                 [--torch] [--no-torch]
                 [--verify] [--pass-pipeline '...']
                 < dev_ir.json > module.mlir
```

Defaults: `--tensor`, `--prefer-affine`, `--torch`.

---

# 11) Worked example (from your snippet)

**Input JSON (abridged)**

```json
{
  "version": 1,
  "module": {
    "name": "helion_module",
    "funcs": [{
      "name": "main",
      "args": [
        {"id":"x","type":{"tensor":{"shape":["?","?"],"elem":"f32"}}},
        {"id":"y","type":{"tensor":{"shape":["?","?"],"elem":"f32"}}}
      ],
      "rets":[{"type":{"tensor":{"shape":["?","?"],"elem":"f32"}}}],
      "body":[
        {"op":"hl.assert_eq","lhs":"dim(x,1)","rhs":"dim(y,0)","msg":"size mismatch"},
        {"op":"hl.tile.begin","iters":["m","n"],"sizes":[64,64],"result":["tm","tn"]},
          {"op":"hl.zeros","shape":["size(tm)","size(tn)"],"dtype":"f32","result":"acc"},
          {"op":"hl.tile.begin","iters":["k"],"sizes":[64],"result":["tk"]},
            {"op":"torch.addmm","args":[ "acc",
              {"slice":{"base":"x","offsets":["tm","tk"],"sizes":["size(tm)","size(tk)"]}},
              {"slice":{"base":"y","offsets":["tk","tn"],"sizes":["size(tk)","size(tn)"]}}
            ],"attrs":{"alpha":1.0,"beta":1.0},"result":"acc"},
          {"op":"hl.tile.end"},
          {"op":"hl.call","callee":"epilogue","args":["acc","tm","tn"],"result":"tile_out"},
          {"op":"hl.store_slice","dst":"out","offsets":["tm","tn"],"src":"tile_out"},
        {"op":"hl.tile.end"},
        {"op":"hl.return","values":["out"]}
      ]
    }]
  }
}
```

**Output MLIR (tensor form, abridged)**

```mlir
module {
  func.func @main(%x: tensor<?x?xf32>, %y: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %M = tensor.dim %x, %c0 : tensor<?x?xf32>
    %K1 = tensor.dim %x, %c1 : tensor<?x?xf32>
    %K2 = tensor.dim %y, %c0 : tensor<?x?xf32>
    %N = tensor.dim %y, %c1 : tensor<?x?xf32>
    %ok = arith.cmpi eq, %K1, %K2 : index
    cf.assert %ok, "size mismatch" : i1

    %out = tensor.empty(%M, %N) : tensor<?x?xf32>

    // Outer tiles (prefer affine; if Tm/Tn dynamic -> scf.for)
    affine.for %im = 0 to %M step 64 {
      %ms = affine.min affine_map<(d0)[s0] -> (64, s0 - d0)>(%im)[%M]
      affine.for %in = 0 to %N step 64 {
        %ns = affine.min affine_map<(d0)[s0] -> (64, s0 - d0)>(%in)[%N]

        %acc = tensor.empty(%ms, %ns) : tensor<?x?xf32>
        %zero = arith.constant 0.0 : f32
        %acc0 = linalg.fill ins(%zero : f32) outs(%acc : tensor<?x?xf32>) -> tensor<?x?xf32>

        affine.for %ik = 0 to %K1 step 64 {
          %ks = affine.min affine_map<(d0)[s0] -> (64, s0 - d0)>(%ik)[%K1]
          %xT = tensor.extract_slice %x[%im, %ik] [%ms, %ks] [1, 1]
                 : tensor<?x?xf32> to tensor<?x?xf32>
          %yT = tensor.extract_slice %y[%ik, %in] [%ks, %ns] [1, 1]
                 : tensor<?x?xf32> to tensor<?x?xf32>

          %acc1 = "torch.aten.addmm"(%acc0, %xT, %yT)
                   {alpha = 1.0 : f64, beta = 1.0 : f64}
                   : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
          %acc0 = %acc1 : tensor<?x?xf32>
        }

        %tile_out = func.call @epilogue(%acc0, %im, %in)
                      : (tensor<?x?xf32>, index, index) -> tensor<?x?xf32>

        %out_t = tensor.extract_slice %out[%im, %in] [%ms, %ns] [1,1]
                  : tensor<?x?xf32> to tensor<?x?xf32>
        %out_t2 = "some.copy"(%tile_out) : (tensor<?x?xf32>) -> tensor<?x?xf32> // replace with real copy
        %out2 = tensor.insert_slice %out_t2 into %out[%im, %in] [%ms, %ns] [1,1]
                 : tensor<?x?xf32> into tensor<?x?xf32>
        %out = %out2 : tensor<?x?xf32>
      }
    }
    return %out : tensor<?x?xf32>
  }
  func.func @epilogue(%t: tensor<?x?xf32>, %i: index, %j: index) -> tensor<?x?xf32> attributes { llvm.emit_c_interface }
}
```

(The example uses `affine.for` + `affine.min` and keeps math in `torch` ops, as requested. `cf.assert` models the Python `assert`. ([mlir.llvm.org][1]))

---

# 12) Edge cases & policies

* **Dynamic tile sizes** ➜ use `scf.for` (since `affine.for` step must be constant). ([mlir.llvm.org][10])
* **Non-affine indexing inside tiles** ➜ force `scf`.
* **Parallel tiles** ➜ optional `scf.forall` front (future switch), later convertible to `scf.for`/`scf.parallel` via standard transforms. ([mlir.llvm.org][8])
* **Tensor vs memref** ➜ prefer **tensor** at the JSON boundary; add a `--memref` flag if you must match an existing runtime. Bufferization is standard. ([mlir.llvm.org][7])

---

# 13) Verification & docs your team will lean on

* `scf` dialect (`scf.for`, `scf.forall`). ([mlir.llvm.org][8])
* `affine` dialect (`affine.for`, `affine.min`, legality & affine maps). ([mlir.llvm.org][1])
* `tensor` & `memref` (slices: `extract_slice`/`subview`). ([mlir.llvm.org][4])
* `cf.assert` for runtime assertions. ([mlir.llvm.org][6])
* Passes catalog (for optional clean-ups). ([mlir.llvm.org][9])
* Torch-MLIR project (for the `torch` dialect ops you want to keep). ([GitHub][3])

---

# 14) Definition-of-Done (practical checklist)

* [ ] `helion2json` emits validated JSON for your sample kernels (matmul-like, multi-tile, edge tiles).
* [ ] `json2mlir` generates MLIR that:

  * [ ] contains `affine.for` + `affine.min` when tile steps are static and accesses affine;
  * [ ] falls back to `scf.for` otherwise;
  * [ ] represents math via `torch.aten.*` ops.
* [ ] `mlir-opt -verify-diagnostics` passes on all outputs.
* [ ] FileCheck tests cover: dynamic shapes, mismatched K assert, 1D/2D tiles, dynamic tile fallback, slice correctness.
* [ ] CI builds against LLVM/MLIR + Torch-MLIR; runs `-canonicalize -cse`.

---

If you want, I can also drop in minimal **starter code** for `JsonLoader` and the **tile-lowering** utility so your team can copy-paste into `Builder.cpp`.

[1]: https://mlir.llvm.org/docs/Dialects/Affine/?utm_source=chatgpt.com "'affine' Dialect - MLIR - LLVM"
[2]: https://mlir.llvm.org/docs/?utm_source=chatgpt.com "Code Documentation - MLIR - LLVM"
[3]: https://github.com/llvm/torch-mlir?utm_source=chatgpt.com "The Torch-MLIR project aims to provide first class support ..."
[4]: https://mlir.llvm.org/docs/Dialects/TensorOps/?utm_source=chatgpt.com "'tensor' Dialect - MLIR"
[5]: https://mlir.llvm.org/docs/Dialects/?utm_source=chatgpt.com "Dialects - MLIR"
[6]: https://mlir.llvm.org/docs/Dialects/ControlFlowDialect/?utm_source=chatgpt.com "'cf' Dialect - MLIR - LLVM"
[7]: https://mlir.llvm.org/docs/Bufferization/?utm_source=chatgpt.com "Bufferization - MLIR - LLVM"
[8]: https://mlir.llvm.org/docs/Dialects/SCFDialect/?utm_source=chatgpt.com "'scf' Dialect - MLIR - LLVM"
[9]: https://mlir.llvm.org/docs/Passes/?utm_source=chatgpt.com "Passes - MLIR - LLVM"
[10]: https://mlir.llvm.org/doxygen/AffineOps_8cpp_source.html?utm_source=chatgpt.com "lib/Dialect/Affine/IR/AffineOps.cpp Source File - MLIR"
