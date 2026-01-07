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

def Helion_AllocLikeOp : Helion_Op<"alloc_like",
    [SameOperandsAndResultType]> {
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
// helion.annotate_tensor
//===----------------------------------------------------------------------===//

def Helion_AnnotateTensorOp : Helion_Op<"annotate_tensor"> {
  let summary = "attach Helion metadata to a tensor";
  let description = [{
    The `helion.annotate_tensor` op records Helion bookkeeping such as tensor
    names and dimension SSA values.  It has no results and is intended purely
    for metadata propagation.
  }];
  let arguments = (ins AnyTensor:$value,
                       Index:$dim0,
                       Index:$dim1,
                       StrAttr:$name);
  let results = (outs);
}

//===----------------------------------------------------------------------===//
// helion.load_tile_dynamic
//===----------------------------------------------------------------------===//

def Helion_LoadTileDynamicOp : Helion_Op<"load_tile_dynamic"> {
  let summary = "load a dynamically sized tensor tile";
  let description = [{
    Loads a tile slice from the source tensor, parameterised by dynamic tile
    sizes and optional metadata attributes.
  }];
  let arguments = (ins AnyTensor:$source,
                       Index:$size_m,
                       Index:$size_n,
                       OptionalAttr<ArrayAttr>:$tile,
                       OptionalAttr<ArrayAttr>:$sizes,
                       OptionalAttr<StrAttr>:$tensor_meta,
                       OptionalAttr<StrAttr>:$fx_node);
  let results = (outs AnyTensor:$result);
}

//===----------------------------------------------------------------------===//
// helion.call_torch
//===----------------------------------------------------------------------===//

def Helion_CallTorchOp : Helion_Op<"call_torch", [SameOperandsAndResultType]> {
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

def Helion_PhiOp : Helion_Op<"phi",
    [SameOperandsAndResultType]> {
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
// helion.store_tile_dynamic
//===----------------------------------------------------------------------===//

def Helion_StoreTileDynamicOp : Helion_Op<"store_tile_dynamic"> {
  let summary = "store a dynamically sized tensor tile";
  let description = [{
    Stores the tile back into the destination tensor, threading the dynamic
    tile sizes as operands and preserving metadata for debugging.
  }];
  let arguments = (ins AnyTensor:$destination,
                       AnyTensor:$tile_value,
                       Index:$size_m,
                       Index:$size_n,
                       OptionalAttr<ArrayAttr>:$tile,
                       OptionalAttr<ArrayAttr>:$sizes,
                       OptionalAttr<StrAttr>:$tensor_meta,
                       OptionalAttr<StrAttr>:$fx_node);
  let results = (outs);
}

//===----------------------------------------------------------------------===//
// helion.get_module_attribute
//===----------------------------------------------------------------------===//

def Helion_GetModuleAttributeOp : Helion_Op<"get_module_attribute", [Pure]> {
  let summary = "get a module attribute value";
  let description = [{
    The `helion.get_module_attribute` op retrieves a named attribute from the
    parent module and returns it as an SSA value.  This is used to access
    compile-time parameters (e.g., tile sizes) stored as module attributes.
  }];
  let arguments = (ins StrAttr:$attr_name);
  let results = (outs Index:$result);
}

#endif // HELION_IR_HELIONOPS_TD
