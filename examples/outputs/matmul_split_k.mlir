=== Device IR ===
Graph 0: IfGraphInfo
opcode         name            target                                     args                                                        kwargs
-------------  --------------  -----------------------------------------  ----------------------------------------------------------  --------
placeholder    arg0_1          arg0_1                                     ()                                                          {}
call_function  _new_var        <function _new_var at 0x7f50297c80d0>      (arg0_1,)                                                   {}
call_function  out             <function _host_tensor at 0x7f50297967a0>  ('out',)                                                    {}
call_function  sym_size_int    aten.sym_size.int                          (arg0_1, 0)                                                 {}
call_function  sym_size_int_1  aten.sym_size.int                          (arg0_1, 1)                                                 {}
call_function  atomic_add      <function atomic_add at 0x7f50341c1e10>    (out, [sym_size_int, sym_size_int_1], _new_var, 'relaxed')  {}
output         output          output                                     ([],)                                                       {}
Graph 1: RootGraphInfo
opcode         name          target                                     args                                           kwargs
-------------  ------------  -----------------------------------------  ---------------------------------------------  --------
call_function  a             <function _host_tensor at 0x7f50297967a0>  ('a',)                                         {}
call_function  block_size_0  <function _get_symnode at 0x7f5029795b40>  ('block_size_0',)                              {}
call_function  block_size_2  <function _get_symnode at 0x7f5029795b40>  ('block_size_2',)                              {}
call_function  load          <function load at 0x7f501aa52320>          (a, [block_size_0, block_size_2], None, None)  {}
call_function  b             <function _host_tensor at 0x7f50297967a0>  ('b',)                                         {}
call_function  block_size_1  <function _get_symnode at 0x7f5029795b40>  ('block_size_1',)                              {}
call_function  load_1        <function load at 0x7f501aa52320>          (b, [block_size_2, block_size_1], None, None)  {}
call_function  acc           aten.mm.default                            (load, load_1)                                 {}
call_function  tile_begin    <function tile_begin at 0x7f501aa6dea0>    (block_size_2,)                                {}
call_function  eq_2          <built-in function eq>                     (tile_begin, 0)                                {}
call_function  _if           <function _if at 0x7f5029796e60>           (eq_2, 0, [acc])                               {}
output         output        output                                     (None,)                                        {}


=== Nodes with symbols ===
Node arg0_1 : FakeTensor(..., size=(u0, u1))
Node _new_var : FakeTensor(..., size=(u0, u1))
Node out : FakeTensor(..., size=(s97, s20))
Node sym_size_int : u0
Node sym_size_int_1 : u1
Node atomic_add : FakeTensor(..., size=(u0, u1))
Node a : FakeTensor(..., size=(s97, s98))
Node block_size_0 : u0
Node block_size_2 : u2
Node load : FakeTensor(..., size=(u0, u2))
Node b : FakeTensor(..., size=(s52, s20))
Node block_size_1 : u1
Node load_1 : FakeTensor(..., size=(u2, u1))
Node acc : FakeTensor(..., size=(u0, u1))
Node tile_begin : u3
Node eq_2 : Eq(u3, 0)
Node _if : []


=== Compile Environment ===
Block Sizes (3):
  Block 0: Size=s97, Var=u0, Reduction=False, Source=LoopSpecBlockSizeSource()
  Block 1: Size=s20, Var=u1, Reduction=False, Source=LoopSpecBlockSizeSource()
  Block 2: Size=s98, Var=u2, Reduction=False, Source=LoopSpecBlockSizeSource()
Shape Env (8):
  Var s97: 256
  Var s98: 4096
  Var s52: 4096
  Var s20: 256
  Var u0: 64
  Var u1: 64
  Var u2: 64
  Var u3: 8192


=== MLIR Dump ===
module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
  func.func @split_k_matmul(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c4096 = arith.constant 4096 : index
    %c256 = arith.constant 256 : index
    %0 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_m, upper_bound = 256 : index} : () -> index
    %1 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_n, upper_bound = 256 : index} : () -> index
    %2 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_k, upper_bound = 4096 : index} : () -> index
    %3 = arith.ceildivui %c256, %0 : index
    %4 = arith.ceildivui %c256, %1 : index
    %5 = arith.ceildivui %c4096, %2 : index
    affine.parallel (%arg3, %arg4, %arg5) = (0, 0, 0) to (symbol(%3), symbol(%4), symbol(%5)) {
      %6 = arith.muli %arg3, %0 : index
      %7 = arith.muli %arg5, %2 : index
      %subview = memref.subview %arg1[%6, %7] [%0, %2] [1, 1] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
      %8 = bufferization.to_tensor %subview : memref<?x?xf32, strided<[4096, 1], offset: ?>> to tensor<?x?xf32>
      %9 = arith.muli %arg4, %1 : index
      %subview_0 = memref.subview %arg2[%7, %9] [%2, %1] [1, 1] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
      %10 = bufferization.to_tensor %subview_0 : memref<?x?xf32, strided<[256, 1], offset: ?>> to tensor<?x?xf32>
      %11 = tensor.empty(%0, %1) : tensor<?x?xf32>
      %12 = linalg.fill ins(%cst : f32) outs(%11 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %13 = linalg.matmul ins(%8, %10 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%12 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %14 = arith.cmpi eq, %7, %c0 : index
      scf.if %14 {
        %subview_1 = memref.subview %arg0[%6, %9] [%0, %1] [1, 1] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
        "loom.sum"(%subview_1, %13) : (memref<?x?xf32, strided<[256, 1], offset: ?>>, tensor<?x?xf32>) -> ()
      }
    }
    return
  }
}


mlir-opt validation succeeded.

