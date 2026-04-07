=== Device IR ===
Graph 0: ForLoopGraphInfo
opcode         name            target                                     args                                             kwargs
-------------  --------------  -----------------------------------------  -----------------------------------------------  --------
placeholder    arg0_1          arg0_1                                     ()                                               {}
call_function  _new_var        <function _new_var at 0x733b78ef9260>      (arg0_1,)                                        {}
call_function  x               <function _host_tensor at 0x733b78eef6a0>  ('x',)                                           {}
call_function  sym_size_int    aten.sym_size.int                          (arg0_1, 0)                                      {}
call_function  block_size_2    <function _get_symnode at 0x733b78eee5c0>  ('block_size_2',)                                {}
call_function  load            <function load at 0x733ad39b2b60>          (x, [sym_size_int, block_size_2], None, None)    {}
call_function  y               <function _host_tensor at 0x733b78eef6a0>  ('y',)                                           {}
call_function  sym_size_int_1  aten.sym_size.int                          (arg0_1, 1)                                      {}
call_function  load_1          <function load at 0x733ad39b2b60>          (y, [block_size_2, sym_size_int_1], None, None)  {}
call_function  acc             aten.addmm.default                         (_new_var, load, load_1)                         {}
output         output          output                                     ([acc],)                                         {}
Graph 1: RootGraphInfo
opcode         name          target                                     args                                                      kwargs
-------------  ------------  -----------------------------------------  --------------------------------------------------------  --------
call_function  block_size_0  <function _get_symnode at 0x733b78eee5c0>  ('block_size_0',)                                         {}
call_function  block_size_1  <function _get_symnode at 0x733b78eee5c0>  ('block_size_1',)                                         {}
call_function  acc           <function full at 0x733ad3996340>          ([block_size_0, block_size_1], 0.0, torch.float16, None)  {}
call_function  x_size1       <function _get_symnode at 0x733b78eee5c0>  ('x_size1',)                                              {}
call_function  _for_loop     <function _for_loop at 0x733b78eefa60>     (0, [0], [x_size1], [acc])                                {}
call_function  getitem       <built-in function getitem>                (_for_loop, 0)                                            {}
call_function  _phi          <function _phi at 0x733b78ef8040>          (acc, getitem)                                            {}
call_function  out           <function _host_tensor at 0x733b78eef6a0>  ('out',)                                                  {}
call_function  store         <function store at 0x733ad39b2700>         (out, [block_size_0, block_size_1], _phi, None)           {}
output         output        output                                     (None,)                                                   {}


=== Nodes with symbols ===
Node arg0_1 : FakeTensor(..., size=(u1, u2), dtype=torch.float16)
Node _new_var : FakeTensor(..., size=(u1, u2), dtype=torch.float16)
Node x : FakeTensor(..., size=(s77, s27), dtype=torch.float16)
Node sym_size_int : u1
Node block_size_2 : u3
Node load : FakeTensor(..., size=(u1, u3), dtype=torch.float16)
Node y : FakeTensor(..., size=(s17, s94), dtype=torch.float16)
Node sym_size_int_1 : u2
Node load_1 : FakeTensor(..., size=(u3, u2), dtype=torch.float16)
Node acc : FakeTensor(..., size=(u1, u2), dtype=torch.float16)
Node block_size_0 : u1
Node block_size_1 : u2
Node acc : FakeTensor(..., size=(u1, u2), dtype=torch.float16)
Node x_size1 : s27
Node _for_loop : [FakeTensor(..., size=(u1, u2), dtype=torch.float16)]
Node getitem : FakeTensor(..., size=(u1, u2), dtype=torch.float16)
Node _phi : FakeTensor(..., size=(u1, u2), dtype=torch.float16)
Node out : FakeTensor(..., size=(s77, s94), dtype=torch.float16)


=== Compile Environment ===
Block Sizes (3):
  Block 0: Size=s77, Var=u1, Reduction=False, Source=LoopSpecBlockSizeSource()
  Block 1: Size=s94, Var=u2, Reduction=False, Source=LoopSpecBlockSizeSource()
  Block 2: Size=s27, Var=u3, Reduction=False, Source=LoopSpecBlockSizeSource()
Shape Env (7):
  Var s77: 4096
  Var s27: 512
  Var s17: 512
  Var s94: 4096
  Var u1: 64
  Var u2: 64
  Var u3: 64


=== MLIR Dump ===
#map = affine_map<()[s0] -> (512 ceildiv s0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
  func.func @matmul(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_m, upper_bound = 4096 : index} : () -> index
    %1 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_n, upper_bound = 4096 : index} : () -> index
    %2 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_k, upper_bound = 512 : index} : () -> index
    affine.parallel (%arg3, %arg4) = (0, 0) to (4096 ceildiv symbol(%0), 4096 ceildiv symbol(%1)) {
      %3 = tensor.empty(%0, %1) : tensor<?x?xf16>
      %4 = linalg.fill ins(%cst : f16) outs(%3 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %5 = affine.for %arg5 = 0 to #map()[%2] iter_args(%arg6 = %4) -> (tensor<?x?xf16>) {
        %9 = arith.muli %arg3, %0 : index
        %10 = arith.muli %arg5, %2 : index
        %subview_0 = memref.subview %arg0[%9, %10] [%0, %2] [1, 1] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
        %11 = bufferization.to_tensor %subview_0 : memref<?x?xf16, strided<[512, 1], offset: ?>> to tensor<?x?xf16>
        %12 = arith.muli %arg4, %1 : index
        %subview_1 = memref.subview %arg1[%10, %12] [%2, %1] [1, 1] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
        %13 = bufferization.to_tensor %subview_1 : memref<?x?xf16, strided<[4096, 1], offset: ?>> to tensor<?x?xf16>
        %14 = linalg.matmul ins(%11, %13 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%4 : tensor<?x?xf16>) -> tensor<?x?xf16>
        %15 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg6, %14 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%3 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %in_2: f16, %out: f16):
          %16 = arith.addf %in, %in_2 : f16
          linalg.yield %16 : f16
        } -> tensor<?x?xf16>
        affine.yield %15 : tensor<?x?xf16>
      }
      %6 = arith.muli %arg3, %0 : index
      %7 = arith.muli %arg4, %1 : index
      %subview = memref.subview %arg2[%6, %7] [%0, %1] [1, 1] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
      %8 = bufferization.to_buffer %5 : tensor<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
      memref.copy %8, %subview : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
    }
    return
  }
}


mlir-opt validation succeeded.

