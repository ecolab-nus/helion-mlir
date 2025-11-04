#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

: "${LLVM_DIR:=/mnt/fast/llvm-project/mlir/lib/cmake/llvm}"
: "${MLIR_DIR:=/mnt/fast/llvm-project/mlir/lib/cmake/mlir}"
: "${GENERATOR:=Ninja}"

echo "[json2mlir] Configuring with CMake generator: ${GENERATOR}"
cmake -S "${ROOT_DIR}/json2mlir" \
      -B "${BUILD_DIR}" \
      -G "${GENERATOR}" \
      -DLLVM_DIR="${LLVM_DIR}" \
      -DMLIR_DIR="${MLIR_DIR}"

echo "[json2mlir] Building targets via cmake --build"
cmake --build "${BUILD_DIR}" --target json2mlir

echo "[json2mlir] Build artifacts available in ${BUILD_DIR}"
