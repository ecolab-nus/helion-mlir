#ifndef HELION_MLIR_JSON_LOADER_H
#define HELION_MLIR_JSON_LOADER_H

#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/ADT/StringRef.h"

namespace helion_mlir {

/// Parse the provided JSON text and return the top-level value.
/// Returns an llvm::Error with diagnostic information on failure.
llvm::Expected<llvm::json::Value> parseJsonString(llvm::StringRef buffer);

} // namespace helion_mlir

#endif // HELION_MLIR_JSON_LOADER_H
