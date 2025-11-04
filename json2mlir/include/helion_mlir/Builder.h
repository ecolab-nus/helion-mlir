#ifndef HELION_MLIR_BUILDER_H
#define HELION_MLIR_BUILDER_H

#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
class MLIRContext;
class ModuleOp;
} // namespace mlir

namespace helion_mlir {

/// Translate the given JSON value describing a Helion module into an MLIR module.
llvm::Expected<mlir::ModuleOp> buildModuleFromJson(const llvm::json::Value &root,
                                                   mlir::MLIRContext &ctx);

/// Convenience helper that parses the JSON text and creates the MLIR module.
llvm::Expected<mlir::ModuleOp> buildModuleFromJsonString(llvm::StringRef buffer,
                                                         mlir::MLIRContext &ctx);

} // namespace helion_mlir

#endif // HELION_MLIR_BUILDER_H
