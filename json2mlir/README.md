# json2mlir prototype

This directory contains an experimental JSON ➜ MLIR translator that consumes the
interchange format produced by `helion2json`.

## Building

```
cmake -S json2mlir -B build -G Ninja \
      -DLLVM_DIR=/mnt/fast/llvm-project/mlir/lib/cmake/llvm \
      -DMLIR_DIR=/mnt/fast/llvm-project/mlir/lib/cmake/mlir
ninja -C build json2mlir
```

Adjust the `LLVM_DIR`/`MLIR_DIR` paths to point at your local MLIR install.

## Using the CLI

```
build/bin/json2mlir <path/to/input.json> [-o output.mlir]
```

The translator currently emits memref-based MLIR loops (`scf.for`) and treats
`torch.addmm` as an external side-effecting call. A sample input/output pair is
available under `json2mlir/tests/FileCheck/`.
