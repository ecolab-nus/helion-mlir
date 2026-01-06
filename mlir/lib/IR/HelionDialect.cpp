//===- HelionDialect.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the Helion FX → MLIR prototype.
//
//===----------------------------------------------------------------------===//

#include "helion/IR/HelionDialect.h"
#include "helion/IR/HelionOps.h"

using namespace mlir;
using namespace helion;

#include "HelionDialect.cpp.inc"

void HelionDialect::initialize() {
  addOperations<
      AllocLikeOp,
      AnnotateTensorOp,
      CallTorchOp,
      LoadTileDynamicOp,
      PhiOp,
      StoreTileDynamicOp,
      ZeroTileOp
      >();
}
