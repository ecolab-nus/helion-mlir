//===- HelionOps.td -----------------------------------------------------*- tablegen -*-===//
//
// Part of the Helion FX → MLIR prototype.
//
//===----------------------------------------------------------------------===//

#ifndef HELION_IR_HELIONOPS_TD
#define HELION_IR_HELIONOPS_TD

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Dialect definition.
def Helion_Dialect : Dialect {
  let name = "helion";
  let summary = "Helion device IR surface dialect";
  let cppNamespace = "::helion";
}

// Base class for Helion operations.
class Helion_Op<string mnemonic, list<Trait> traits = []> :
    Op<Helion_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// helion.phi
//===----------------------------------------------------------------------===//

def Helion_PhiOp : Helion_Op<"phi"> {
  let summary = "Helion loop-carried value placeholder";
  let description = [{
    Models a phi node that threads the carried tensor across loop iterations.
  }];
  let arguments = (ins AnyTensor:$initial,
                       AnyTensor:$current,
                       OptionalAttr<StrAttr>:$fx_node);
  let results = (outs AnyTensor:$result);
}

//===----------------------------------------------------------------------===//

#endif // HELION_IR_HELIONOPS_TD

