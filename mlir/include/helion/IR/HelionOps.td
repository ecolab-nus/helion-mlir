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
// helion.alloc_like
//===----------------------------------------------------------------------===//

def Helion_AllocLikeOp : Helion_Op<"alloc_like"> {
  let summary = "allocate a tensor like the source operand";
  let description = [{
    The `helion.alloc_like` op materialises a tensor SSA value matching the
    element type of the source operand.  The optional `shape` array attribute
    records the extents that Helion computed for the allocation with `-1`
    denoting a dynamic dimension.
  }];
  let arguments = (ins AnyTensor:$input,
                       OptionalAttr<ArrayAttr>:$shape);
  let results = (outs AnyTensor:$result);
}

//===----------------------------------------------------------------------===//
// helion.zero_tile
//===----------------------------------------------------------------------===//

def Helion_ZeroTileOp : Helion_Op<"zero_tile"> {
  let summary = "create a tensor tile initialised to zero";
  let description = [{
    The `helion.zero_tile` op produces a tile tensor pre-filled with zeros.
    The op carries the logical tile `shape` (using `-1` for dynamic sizes) and
    the destination `dtype`.
  }];
  let arguments = (ins ArrayAttr:$shape,
                       TypeAttr:$dtype);
  let results = (outs AnyTensor:$result);
}

//===----------------------------------------------------------------------===//
// helion.full
//===----------------------------------------------------------------------===//

def Helion_FullOp : Helion_Op<"full"> {
  let summary = "create a tensor tile filled with a constant value";
  let description = [{
    The `helion.full` op produces a tile tensor pre-filled with a constant value.
    The dimensions are passed as index operands. The fill value can be passed:
    - As an optional `fill_value` F64 attribute for normal values
    - As an optional f32 operand for special values like -inf (using arith.constant)
    The `dtype` specifies the result element type.
  }];
  let arguments = (ins Variadic<AnyTypeOf<[Index, AnyFloat]>>:$operands,
                       OptionalAttr<F64Attr>:$fill_value,
                       TypeAttr:$dtype);
  let results = (outs AnyTensor:$result);
}

//===----------------------------------------------------------------------===//
// helion.load
//===----------------------------------------------------------------------===//

def Helion_LoadOp : Helion_Op<"load"> {
  let summary = "load a dynamically sized tensor tile";
  let description = [{
    Loads a tile slice from the source tensor. The `indices` attribute contains
    string representations of the indexing expressions for debugging/analysis.
  }];
  let arguments = (ins AnyTensor:$source,
                       OptionalAttr<ArrayAttr>:$indices,
                       OptionalAttr<StrAttr>:$fx_node);
  let results = (outs AnyTensor:$result);
}

//===----------------------------------------------------------------------===//
// helion.call_torch
//===----------------------------------------------------------------------===//

def Helion_CallTorchOp : Helion_Op<"call_torch"> {
  let summary = "delegate computation to a Torch operator";
  let description = [{
    Proxies Helion math to an underlying Torch operator until the lowering is
    implemented natively in MLIR.
  }];
  let arguments = (ins AnyTensor:$accumulator,
                       AnyTensor:$lhs,
                       AnyTensor:$rhs,
                       StrAttr:$fn_name,
                       OptionalAttr<StrAttr>:$fx_node);
  let results = (outs AnyTensor:$result);
}

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
// helion.store
//===----------------------------------------------------------------------===//

def Helion_StoreOp : Helion_Op<"store"> {
  let summary = "store a dynamically sized tensor tile";
  let description = [{
    Stores the tile back into the destination tensor. The `indices` attribute 
    contains string representations of the indexing expressions for debugging/analysis.
  }];
  let arguments = (ins AnyTensor:$destination,
                       AnyTensor:$tile_value,
                       OptionalAttr<ArrayAttr>:$indices,
                       OptionalAttr<StrAttr>:$fx_node);
  let results = (outs);
}

//===----------------------------------------------------------------------===//
// helion.get_module_attribute
//===----------------------------------------------------------------------===//

def Helion_GetModuleAttributeOp : Helion_Op<"get_module_attribute"> {
  let summary = "get a module attribute value";
  let description = [{
    The `helion.get_module_attribute` op retrieves a named attribute from the
    parent module and returns it as an SSA value.  This is used to access
    compile-time parameters (e.g., tile sizes) stored as module attributes.
  }];
  let arguments = (ins StrAttr:$attr_name);
  let results = (outs Index:$result);
}
//===----------------------------------------------------------------------===//
// helion.host_ref
//===----------------------------------------------------------------------===//

def Helion_HostRefOp : Helion_Op<"host_ref"> {
  let summary = "reference to a derived/intermediate host tensor";
  let description = [{
    The `helion.host_ref` op represents a reference to a tensor that exists on
    the host but is not a direct kernel argument. These are typically view/reshape
    operations of input tensors (e.g., k_view, q_view in attention kernels).
    The op carries a `name` attribute to identify which host tensor is referenced.
  }];
  let arguments = (ins StrAttr:$name,
                       OptionalAttr<StrAttr>:$fx_node);
  let results = (outs AnyTensor:$result);
}

#endif // HELION_IR_HELIONOPS_TD
