=== Device IR ===
Graph 0: IfGraphInfo
opcode         name            target                                     args                                                        kwargs
-------------  --------------  -----------------------------------------  ----------------------------------------------------------  --------
placeholder    arg0_1          arg0_1                                     ()                                                          {}
call_function  _new_var        <function _new_var at 0x7f5154481510>      (arg0_1,)                                                   {}
call_function  _mask_to        <function _mask_to at 0x7f51544811b0>      (_new_var, 0)                                               {}
call_function  acc_across_k    aten.sum.dim_IntList                       (_mask_to, [0])                                             {}
call_function  out_            <function _host_tensor at 0x7f5154447be0>  ('out_',)                                                   {}
call_function  sym_size_int    aten.sym_size.int                          (arg0_1, 1)                                                 {}
call_function  sym_size_int_1  aten.sym_size.int                          (arg0_1, 2)                                                 {}
call_function  store           <function store at 0x7f513b949b40>         (out_, [sym_size_int, sym_size_int_1], acc_across_k, None)  {}
output         output          output                                     ([],)                                                       {}
Graph 1: RootGraphInfo
opcode         name          target                                     args                                           kwargs
-------------  ------------  -----------------------------------------  ---------------------------------------------  --------
call_function  a             <function _host_tensor at 0x7f5154447be0>  ('a',)                                         {}
call_function  block_size_0  <function _get_symnode at 0x7f5154446f80>  ('block_size_0',)                              {}
call_function  block_size_2  <function _get_symnode at 0x7f5154446f80>  ('block_size_2',)                              {}
call_function  load          <function load at 0x7f513b94b760>          (a, [block_size_0, block_size_2], None, None)  {}
call_function  b             <function _host_tensor at 0x7f5154447be0>  ('b',)                                         {}
call_function  block_size_1  <function _get_symnode at 0x7f5154446f80>  ('block_size_1',)                              {}
call_function  load_1        <function load at 0x7f513b94b760>          (b, [block_size_2, block_size_1], None, None)  {}
call_function  local_acc     aten.mm.default                            (load, load_1)                                 {}
call_function  gathered      <function gather at 0x7f513b6edab0>        (block_size_2, local_acc)                      {}
call_function  tile_id       <function tile_id at 0x7f513b967be0>       (block_size_2,)                                {}
call_function  eq_2          <built-in function eq>                     (tile_id, 0)                                   {}
call_function  _if           <function _if at 0x7f5154480310>           (eq_2, 0, [gathered])                          {}
output         output        output                                     (None,)                                        {}


=== Nodes with symbols ===
Node arg0_1 : FakeTensor(..., size=(u2, u0, u1), dtype=torch.float16)
Node _new_var : FakeTensor(..., size=(u2, u0, u1), dtype=torch.float16)
Node _mask_to : FakeTensor(..., size=(u2, u0, u1), dtype=torch.float16)
Node acc_across_k : FakeTensor(..., size=(u0, u1), dtype=torch.float16)
Node out_ : FakeTensor(..., size=(s97, s20), dtype=torch.float16)
Node sym_size_int : u0
Node sym_size_int_1 : u1
Node a : FakeTensor(..., size=(s97, s98), dtype=torch.float16)
Node block_size_0 : u0
Node block_size_2 : u2
Node load : FakeTensor(..., size=(u0, u2), dtype=torch.float16)
Node b : FakeTensor(..., size=(s52, s20), dtype=torch.float16)
Node block_size_1 : u1
Node load_1 : FakeTensor(..., size=(u2, u1), dtype=torch.float16)
Node local_acc : FakeTensor(..., size=(u0, u1), dtype=torch.float16)
Node gathered : FakeTensor(..., size=(u2, u0, u1), dtype=torch.float16)
Node tile_id : u3
Node eq_2 : Eq(u3, 0)
Node _if : []


=== Compile Environment ===
Block Sizes (3):
  Block 0: Size=s97, Var=u0, Reduction=False, Source=LoopSpecBlockSizeSource()
  Block 1: Size=s20, Var=u1, Reduction=False, Source=LoopSpecBlockSizeSource()
  Block 2: Size=s98, Var=u2, Reduction=False, Source=LoopSpecBlockSizeSource()
Shape Env (8):
  Var s97: 512
  Var s98: 4096
  Var s52: 4096
  Var s20: 512
  Var u0: 64
  Var u1: 64
  Var u2: 64
  Var u3: 8192


=== MLIR Dump ===
module attributes {loom.tile_m = {upper_bound = 512 : index, is_reduction = false}, loom.tile_n = {upper_bound = 512 : index, is_reduction = false}, loom.tile_k = {upper_bound = 4096 : index, is_reduction = false}} {
  func.func @split_k_matmul_gather(%out__arg: memref<512x512xf16>, %a_arg: memref<512x4096xf16>, %b_arg: memref<4096x512xf16>) {
    %tile_m = "loom.sym"() {symbol_ref = @tile_m, upper_bound = 512 : index, is_reduction = false} : () -> index
    %tile_n = "loom.sym"() {symbol_ref = @tile_n, upper_bound = 512 : index, is_reduction = false} : () -> index
    %tile_k = "loom.sym"() {symbol_ref = @tile_k, upper_bound = 4096 : index, is_reduction = false} : () -> index
    %loop_extent0 = arith.constant 512 : index
    %trip_count1 = arith.ceildivui %loop_extent0, %tile_m : index
    %loop_extent2 = arith.constant 512 : index
    %trip_count3 = arith.ceildivui %loop_extent2, %tile_n : index
    %loop_extent4 = arith.constant 4096 : index
    %trip_count5 = arith.ceildivui %loop_extent4, %tile_k : index
    affine.parallel (%iv_block_0, %iv_block_1, %iv_block_2) = (0, 0, 0) to (%trip_count1, %trip_count3, %trip_count5) {
      %offset6 = arith.muli %iv_block_0, %tile_m : index
      %offset7 = arith.muli %iv_block_2, %tile_k : index
      %subview8 = memref.subview %a_arg[%offset6, %offset7][%tile_m, %tile_k][1, 1] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
      %tile9 = bufferization.to_tensor %subview8 : memref<?x?xf16, strided<[4096, 1], offset: ?>> to tensor<?x?xf16>
      %offset10 = arith.muli %iv_block_2, %tile_k : index
      %offset11 = arith.muli %iv_block_1, %tile_n : index
      %subview12 = memref.subview %b_arg[%offset10, %offset11][%tile_k, %tile_n][1, 1] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
      %tile13 = bufferization.to_tensor %subview12 : memref<?x?xf16, strided<[512, 1], offset: ?>> to tensor<?x?xf16>
      %t14 = arith.constant 0 : index
      %t15 = arith.constant 1 : index
      %t16 = arith.constant 0.000000e+00 : f16
      %t17 = arith.cmpi eq, %tile_k, %tile_k : index
      cf.assert %t17, "mismatching contracting dimension for torch.aten.mm"
      %t18 = tensor.empty(%tile_m, %tile_n) : tensor<?x?xf16>
      %t19 = linalg.fill ins(%t16 : f16) outs(%t18 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %t20 = linalg.matmul ins(%tile9, %tile13 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%t19 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %t21 = tensor.empty(%tile_m, %tile_n) : tensor<?x?xf16>
      %loop_extent22 = arith.constant 4096 : index
      %trip_count23 = arith.ceildivui %loop_extent22, %tile_k : index
      %gather_out24 = tensor.empty(%trip_count23, %tile_m, %tile_n) : tensor<?x?x?xf16>
      %gathered25 = "loom.gather"(%t20, %gather_out24, %iv_block_2) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0>} : (tensor<?x?xf16>, tensor<?x?x?xf16>, index) -> tensor<?x?x?xf16>
      %cmp_rhs26 = arith.constant 0 : index
      %cmp27 = arith.cmpi eq, %iv_block_2, %cmp_rhs26 : index
      scf.if %cmp27 {
        %t28 = arith.constant 0.000000e+00 : f16
        %t29 = arith.constant 1 : index
        %t30 = arith.constant 2 : index
        %t31 = tensor.empty(%tile_m, %tile_n) : tensor<?x?xf16>
        %t32 = linalg.fill ins(%t28 : f16) outs(%t31 : tensor<?x?xf16>) -> tensor<?x?xf16>
        %t33 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%gathered25 : tensor<?x?x?xf16>) outs(%t32 : tensor<?x?xf16>) {
        ^bb0(%blk_arg34: f16, %blk_arg35: f16):
        %t36 = arith.addf %blk_arg34, %blk_arg35 : f16
        linalg.yield %t36 : f16
        } -> tensor<?x?xf16>
        %offset37 = arith.muli %iv_block_0, %tile_m : index
        %offset38 = arith.muli %iv_block_1, %tile_n : index
        %subview39 = memref.subview %out__arg[%offset37, %offset38][%tile_m, %tile_n][1, 1] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
        %value_memref40 = bufferization.to_buffer %t33 : tensor<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
        memref.copy %value_memref40, %subview39 : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16, strided<[512, 1], offset: ?>>
      }
      affine.yield
    }
    return
  }
}

mlir-opt validation succeeded.

✓  loom.gather op found in MLIR output.
