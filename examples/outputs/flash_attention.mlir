=== Device IR ===
Graph 0: ForLoopGraphInfo
opcode         name            target                                     args                                                                         kwargs
-------------  --------------  -----------------------------------------  ---------------------------------------------------------------------------  --------
placeholder    arg0_1          arg0_1                                     ()                                                                           {}
placeholder    arg1_1          arg1_1                                     ()                                                                           {}
placeholder    arg2_1          arg2_1                                     ()                                                                           {}
placeholder    arg3_1          arg3_1                                     ()                                                                           {}
placeholder    arg4_1          arg4_1                                     ()                                                                           {}
call_function  _new_var        <function _new_var at 0x7f9d53e293f0>      (arg0_1,)                                                                    {}
call_function  _new_var_1      <function _new_var at 0x7f9d53e293f0>      (arg1_1,)                                                                    {}
call_function  _new_var_2      <function _new_var at 0x7f9d53e293f0>      (arg2_1,)                                                                    {}
call_function  _new_var_3      <function _new_var at 0x7f9d53e293f0>      (arg3_1,)                                                                    {}
call_function  _new_var_4      <function _new_var at 0x7f9d53e293f0>      (arg4_1,)                                                                    {}
call_function  k_view          <function _host_tensor at 0x7f9d53df7ac0>  ('k_view',)                                                                  {}
call_function  sym_size_int    aten.sym_size.int                          (arg0_1, 0)                                                                  {}
call_function  block_size_3    <function _get_symnode at 0x7f9d53df6e60>  ('block_size_3',)                                                            {}
call_function  k               <function load at 0x7f9d452b3640>          (k_view, [sym_size_int, slice(None, None, None), block_size_3], None, None)  {}
call_function  sym_size_int_1  aten.sym_size.int                          (arg0_1, 1)                                                                  {}
call_function  qk              aten.bmm.default                           (_new_var, k)                                                                {}
call_function  _mask_to_2      <function _mask_to at 0x7f9d53e29090>      (qk, -inf)                                                                   {}
call_function  amax            aten.amax.default                          (_mask_to_2, [-1], True)                                                     {}
call_function  mul             aten.mul.Tensor                            (amax, _new_var_2)                                                           {}
call_function  m_ij            aten.maximum.default                       (_new_var_1, mul)                                                            {}
call_function  m_ij_broad      <function broadcast at 0x7f9d4507d7e0>     (m_ij, 2, [sym_size_int, sym_size_int_1, block_size_3])                      {}
call_function  mul_1           aten.mul.Tensor                            (qk, _new_var_2)                                                             {}
call_function  qk_1            aten.sub.Tensor                            (mul_1, m_ij_broad)                                                          {}
call_function  exp             aten.exp.default                           (qk_1,)                                                                      {}
call_function  _mask_to_3      <function _mask_to at 0x7f9d53e29090>      (exp, 0)                                                                     {}
call_function  l_ij            aten.sum.dim_IntList                       (_mask_to_3, [-1], True)                                                     {}
call_function  sub_1           aten.sub.Tensor                            (_new_var_1, m_ij)                                                           {}
call_function  alpha           aten.exp.default                           (sub_1,)                                                                     {}
call_function  mul_2           aten.mul.Tensor                            (_new_var_3, alpha)                                                          {}
call_function  l_i             aten.add.Tensor                            (mul_2, l_ij)                                                                {}
call_function  acc             aten.mul.Tensor                            (_new_var_4, alpha)                                                          {}
call_function  v_view          <function _host_tensor at 0x7f9d53df7ac0>  ('v_view',)                                                                  {}
call_function  v               <function load at 0x7f9d452b3640>          (v_view, [sym_size_int, block_size_3, slice(None, None, None)], None, None)  {}
call_function  acc_1           aten.baddbmm.default                       (acc, _mask_to_3, v)                                                         {}
call_function  m_i             <function _new_var at 0x7f9d53e293f0>      (m_ij,)                                                                      {}
output         output          output                                     ([m_i, l_i, acc_1],)                                                         {}
Graph 1: RootGraphInfo
opcode         name           target                                     args                                                                         kwargs
-------------  -------------  -----------------------------------------  ---------------------------------------------------------------------------  ----------------------------------------------------------------------------------------------------
call_function  qk_scale_dev   <function full at 0x7f9d452a2e60>          ([], 0.08838834764831843, torch.float16, None)                               {}
call_function  block_size_0   <function _get_symnode at 0x7f9d53df6e60>  ('block_size_0',)                                                            {}
call_function  block_size_1   <function _get_symnode at 0x7f9d53df6e60>  ('block_size_1',)                                                            {}
call_function  m_i            <function full at 0x7f9d452a2e60>          ([block_size_0, block_size_1, 1], -inf, torch.float16, None)                 {}
call_function  l_i            aten.full.default                          ([block_size_0, block_size_1, 1], 1.0)                                       {'dtype': torch.float16, 'layout': torch.strided, 'device': device(type='cpu'), 'pin_memory': False}
call_function  acc            <function full at 0x7f9d452a2e60>          ([block_size_0, block_size_1, 128], 0.0, torch.float16, None)                {}
call_function  q_view         <function _host_tensor at 0x7f9d53df7ac0>  ('q_view',)                                                                  {}
call_function  q              <function load at 0x7f9d452b3640>          (q_view, [block_size_0, block_size_1, slice(None, None, None)], None, None)  {}
call_function  k_in_size1     <function _get_symnode at 0x7f9d53df6e60>  ('k_in_size1',)                                                              {}
call_function  _for_loop      <function _for_loop at 0x7f9d53df7e20>     (0, [0], [k_in_size1], [q, m_i, qk_scale_dev, l_i, acc])                     {}
call_function  getitem        <built-in function getitem>                (_for_loop, 0)                                                               {}
call_function  getitem_1      <built-in function getitem>                (_for_loop, 1)                                                               {}
call_function  getitem_2      <built-in function getitem>                (_for_loop, 2)                                                               {}
call_function  _phi           <function _phi at 0x7f9d53e283a0>          (m_i, getitem)                                                               {}
call_function  _phi_1         <function _phi at 0x7f9d53e283a0>          (l_i, getitem_1)                                                             {}
call_function  _phi_2         <function _phi at 0x7f9d53e283a0>          (acc, getitem_2)                                                             {}
call_function  l_i_broadcast  <function broadcast at 0x7f9d4507d7e0>     (_phi_1, 2, [block_size_0, block_size_1, 128])                               {}
call_function  acc_1          aten.div.Tensor                            (_phi_2, l_i_broadcast)                                                      {}
call_function  out_           <function _host_tensor at 0x7f9d53df7ac0>  ('out_',)                                                                    {}
call_function  store          <function store at 0x7f9d452b1a20>         (out_, [block_size_0, block_size_1, slice(None, None, None)], acc_1, None)   {}
output         output         output                                     (None,)                                                                      {}


=== Nodes with symbols ===
Node arg0_1 : FakeTensor(..., size=(u4, u5, 128), dtype=torch.float16)
Node arg1_1 : FakeTensor(..., size=(u4, u5, 1), dtype=torch.float16)
Node arg2_1 : FakeTensor(..., size=(), dtype=torch.float16)
Node arg3_1 : FakeTensor(..., size=(u4, u5, 1), dtype=torch.float16)
Node arg4_1 : FakeTensor(..., size=(u4, u5, 128), dtype=torch.float16)
Node _new_var : FakeTensor(..., size=(u4, u5, 128), dtype=torch.float16)
Node _new_var_1 : FakeTensor(..., size=(u4, u5, 1), dtype=torch.float16)
Node _new_var_2 : FakeTensor(..., size=(), dtype=torch.float16)
Node _new_var_3 : FakeTensor(..., size=(u4, u5, 1), dtype=torch.float16)
Node _new_var_4 : FakeTensor(..., size=(u4, u5, 128), dtype=torch.float16)
Node k_view : FakeTensor(..., size=(s35, 128, s34), dtype=torch.float16)
Node sym_size_int : u4
Node block_size_3 : u7
Node k : FakeTensor(..., size=(u4, 128, u7), dtype=torch.float16)
Node sym_size_int_1 : u5
Node qk : FakeTensor(..., size=(u4, u5, u7), dtype=torch.float16)
Node _mask_to_2 : FakeTensor(..., size=(u4, u5, u7), dtype=torch.float16)
Node amax : FakeTensor(..., size=(u4, u5, 1), dtype=torch.float16)
Node mul : FakeTensor(..., size=(u4, u5, 1), dtype=torch.float16)
Node m_ij : FakeTensor(..., size=(u4, u5, 1), dtype=torch.float16)
Node m_ij_broad : FakeTensor(..., size=(u4, u5, u7), dtype=torch.float16)
Node mul_1 : FakeTensor(..., size=(u4, u5, u7), dtype=torch.float16)
Node qk_1 : FakeTensor(..., size=(u4, u5, u7), dtype=torch.float16)
Node exp : FakeTensor(..., size=(u4, u5, u7), dtype=torch.float16)
Node _mask_to_3 : FakeTensor(..., size=(u4, u5, u7), dtype=torch.float16)
Node l_ij : FakeTensor(..., size=(u4, u5, 1), dtype=torch.float16)
Node sub_1 : FakeTensor(..., size=(u4, u5, 1), dtype=torch.float16)
Node alpha : FakeTensor(..., size=(u4, u5, 1), dtype=torch.float16)
Node mul_2 : FakeTensor(..., size=(u4, u5, 1), dtype=torch.float16)
Node l_i : FakeTensor(..., size=(u4, u5, 1), dtype=torch.float16)
Node acc : FakeTensor(..., size=(u4, u5, 128), dtype=torch.float16)
Node v_view : FakeTensor(..., size=(s80, s34, 128), dtype=torch.float16)
Node v : FakeTensor(..., size=(u4, u7, 128), dtype=torch.float16)
Node acc_1 : FakeTensor(..., size=(u4, u5, 128), dtype=torch.float16)
Node m_i : FakeTensor(..., size=(u4, u5, 1), dtype=torch.float16)
Node qk_scale_dev : FakeTensor(..., size=(), dtype=torch.float16)
Node block_size_0 : u4
Node block_size_1 : u5
Node m_i : FakeTensor(..., size=(u4, u5, 1), dtype=torch.float16)
Node l_i : FakeTensor(..., size=(u4, u5, 1), dtype=torch.float16)
Node acc : FakeTensor(..., size=(u4, u5, 128), dtype=torch.float16)
Node q_view : FakeTensor(..., size=(s30, s48, 128), dtype=torch.float16)
Node q : FakeTensor(..., size=(u4, u5, 128), dtype=torch.float16)
Node k_in_size1 : s34
Node _for_loop : [FakeTensor(..., size=(u4, u5, 1), dtype=torch.float16), FakeTensor(..., size=(u4, u5, 1), dtype=torch.float16), FakeTensor(..., size=(u4, u5, 128), dtype=torch.float16)]
Node getitem : FakeTensor(..., size=(u4, u5, 1), dtype=torch.float16)
Node getitem_1 : FakeTensor(..., size=(u4, u5, 1), dtype=torch.float16)
Node getitem_2 : FakeTensor(..., size=(u4, u5, 128), dtype=torch.float16)
Node _phi : FakeTensor(..., size=(u4, u5, 1), dtype=torch.float16)
Node _phi_1 : FakeTensor(..., size=(u4, u5, 1), dtype=torch.float16)
Node _phi_2 : FakeTensor(..., size=(u4, u5, 128), dtype=torch.float16)
Node l_i_broadcast : FakeTensor(..., size=(u4, u5, 128), dtype=torch.float16)
Node acc_1 : FakeTensor(..., size=(u4, u5, 128), dtype=torch.float16)
Node out_ : FakeTensor(..., size=(s30, s48, 128), dtype=torch.float16)
Node arg0_1 : FakeTensor(..., size=(u4, u5, 128), dtype=torch.float16)
Node arg1_1 : FakeTensor(..., size=(u4, u5, 1), dtype=torch.float16)
Node arg2_1 : FakeTensor(..., size=(), dtype=torch.float16)
Node arg3_1 : FakeTensor(..., size=(u4, u5, 1), dtype=torch.float16)
Node arg4_1 : FakeTensor(..., size=(u4, u5, 128), dtype=torch.float16)
Node _new_var : FakeTensor(..., size=(u4, u5, 128), dtype=torch.float16)
Node _new_var_1 : FakeTensor(..., size=(u4, u5, 1), dtype=torch.float16)
Node _new_var_2 : FakeTensor(..., size=(), dtype=torch.float16)
Node _new_var_3 : FakeTensor(..., size=(u4, u5, 1), dtype=torch.float16)
Node _new_var_4 : FakeTensor(..., size=(u4, u5, 128), dtype=torch.float16)
Node k_view : FakeTensor(..., size=(s35, 128, s34), dtype=torch.float16)
Node sym_size_int : u4
Node block_size_3 : u7
Node k : FakeTensor(..., size=(u4, 128, u7), dtype=torch.float16)
Node sym_size_int_1 : u5
Node qk : FakeTensor(..., size=(u4, u5, u7), dtype=torch.float16)
Node _mask_to_2 : FakeTensor(..., size=(u4, u5, u7), dtype=torch.float16)
Node amax : FakeTensor(..., size=(u4, u5, 1), dtype=torch.float16)
Node mul : FakeTensor(..., size=(u4, u5, 1), dtype=torch.float16)
Node m_ij : FakeTensor(..., size=(u4, u5, 1), dtype=torch.float16)
Node m_ij_broad : FakeTensor(..., size=(u4, u5, u7), dtype=torch.float16)
Node mul_1 : FakeTensor(..., size=(u4, u5, u7), dtype=torch.float16)
Node qk_1 : FakeTensor(..., size=(u4, u5, u7), dtype=torch.float16)
Node exp : FakeTensor(..., size=(u4, u5, u7), dtype=torch.float16)
Node _mask_to_3 : FakeTensor(..., size=(u4, u5, u7), dtype=torch.float16)
Node l_ij : FakeTensor(..., size=(u4, u5, 1), dtype=torch.float16)
Node sub_1 : FakeTensor(..., size=(u4, u5, 1), dtype=torch.float16)
Node alpha : FakeTensor(..., size=(u4, u5, 1), dtype=torch.float16)
Node mul_2 : FakeTensor(..., size=(u4, u5, 1), dtype=torch.float16)
Node l_i : FakeTensor(..., size=(u4, u5, 1), dtype=torch.float16)
Node acc : FakeTensor(..., size=(u4, u5, 128), dtype=torch.float16)
Node v_view : FakeTensor(..., size=(s80, s34, 128), dtype=torch.float16)
Node v : FakeTensor(..., size=(u4, u7, 128), dtype=torch.float16)
Node acc_1 : FakeTensor(..., size=(u4, u5, 128), dtype=torch.float16)
Node m_i : FakeTensor(..., size=(u4, u5, 1), dtype=torch.float16)
Node qk_scale_dev : FakeTensor(..., size=(), dtype=torch.float16)
Node block_size_0 : u4
Node block_size_1 : u5
Node m_i : FakeTensor(..., size=(u4, u5, 1), dtype=torch.float16)
Node l_i : FakeTensor(..., size=(u4, u5, 1), dtype=torch.float16)
Node acc : FakeTensor(..., size=(u4, u5, 128), dtype=torch.float16)
Node q_view : FakeTensor(..., size=(s30, s48, 128), dtype=torch.float16)
Node q : FakeTensor(..., size=(u4, u5, 128), dtype=torch.float16)
Node k_in_size1 : s34
Node _for_loop : [FakeTensor(..., size=(u4, u5, 1), dtype=torch.float16), FakeTensor(..., size=(u4, u5, 1), dtype=torch.float16), FakeTensor(..., size=(u4, u5, 128), dtype=torch.float16)]
Node getitem : FakeTensor(..., size=(u4, u5, 1), dtype=torch.float16)
Node getitem_1 : FakeTensor(..., size=(u4, u5, 1), dtype=torch.float16)
Node getitem_2 : FakeTensor(..., size=(u4, u5, 128), dtype=torch.float16)
Node _phi : FakeTensor(..., size=(u4, u5, 1), dtype=torch.float16)
Node _phi_1 : FakeTensor(..., size=(u4, u5, 1), dtype=torch.float16)
Node _phi_2 : FakeTensor(..., size=(u4, u5, 128), dtype=torch.float16)
Node l_i_broadcast : FakeTensor(..., size=(u4, u5, 128), dtype=torch.float16)
Node acc_1 : FakeTensor(..., size=(u4, u5, 128), dtype=torch.float16)
Node out_ : FakeTensor(..., size=(s30, s48, 128), dtype=torch.float16)


=== Compile Environment ===
Block Sizes (4):
  Block 0: Size=s30, Var=u4, Reduction=False, Source=LoopSpecBlockSizeSource()
  Block 1: Size=s48, Var=u5, Reduction=False, Source=LoopSpecBlockSizeSource()
  Block 2: Size=128, Var=128, Reduction=True, Source=ReductionLoopBlockSizeSource(reduction_loop=0)
  Block 3: Size=s34, Var=u7, Reduction=False, Source=LoopSpecBlockSizeSource()
Shape Env (13):
  Var s30: 32
  Var s48: 4096
  Var s22: 128
  Var s35: 32
  Var s34: 4096
  Var s4: 128
  Var s80: 32
  Var s41: 4096
  Var s66: 128
  Var u4: 64
  Var u5: 64
  Var u6: 128
  Var u7: 64


=== MLIR Dump ===
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
module attributes {loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
  func.func @attention(%k_view_arg: memref<32x128x4096xf16>, %v_view_arg: memref<32x4096x128xf16>, %q_view_arg: memref<32x4096x128xf16>, %out__arg: memref<32x4096x128xf16>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %cst_0 = arith.constant 1.000000e+00 : f16
    %cst_1 = arith.constant 0xFC00 : f16
    %c1 = arith.constant 1 : index
    %cst_2 = arith.constant 8.837890e-02 : f16
    %c32 = arith.constant 32 : index
    %c4096 = arith.constant 4096 : index
    %0 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_b, upper_bound = 32 : index} : () -> index
    %1 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_m, upper_bound = 4096 : index} : () -> index
    %2 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_n, upper_bound = 4096 : index} : () -> index
    %3 = arith.ceildivui %c32, %0 : index
    %4 = arith.ceildivui %c4096, %1 : index
    affine.parallel (%arg4, %arg5) = (0, 0) to (symbol(%3), symbol(%4)) {
      %5 = tensor.empty(%0, %1) : tensor<?x?x1xf16>
      %6 = linalg.fill ins(%cst_1 : f16) outs(%5 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
      %7 = linalg.fill ins(%cst_0 : f16) outs(%5 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
      %8 = tensor.empty(%0, %1) : tensor<?x?x128xf16>
      %9 = linalg.fill ins(%cst : f16) outs(%8 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
      %10 = arith.muli %arg4, %0 : index
      %11 = arith.muli %arg5, %1 : index
      %subview = memref.subview %q_view_arg[%10, %11, 0] [%0, %1, 128] [1, 1, 1] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
      %12 = bufferization.to_tensor %subview : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to tensor<?x?x128xf16>
      %13 = arith.ceildivui %c4096, %2 : index
      %14:3 = scf.for %arg6 = %c0 to %13 step %c1 iter_args(%arg7 = %6, %arg8 = %7, %arg9 = %9) -> (tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>) {
        %19 = arith.muli %arg6, %2 : index
        %subview_4 = memref.subview %k_view_arg[%10, 0, %19] [%0, 128, %2] [1, 1, 1] : memref<32x128x4096xf16> to memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>>
        %20 = bufferization.to_tensor %subview_4 : memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>> to tensor<?x128x?xf16>
        %21 = arith.index_cast %0 : index to i64
        %22 = arith.cmpi eq, %21, %21 : i64
        cf.assert %22, "mismatching contracting dimension"
        %23 = tensor.empty(%0, %1, %2) : tensor<?x?x?xf16>
        %24 = linalg.fill ins(%cst : f16) outs(%23 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
        %25 = linalg.batch_matmul ins(%12, %20 : tensor<?x?x128xf16>, tensor<?x128x?xf16>) outs(%24 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
        %26 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%25 : tensor<?x?x?xf16>) outs(%6 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %46 = arith.maximumf %in, %out : f16
          linalg.yield %46 : f16
        } -> tensor<?x?x1xf16>
        %27 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26 : tensor<?x?x1xf16>) outs(%5 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %46 = arith.mulf %in, %cst_2 : f16
          linalg.yield %46 : f16
        } -> tensor<?x?x1xf16>
        %28 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg7, %27 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%5 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %in_6: f16, %out: f16):
          %46 = arith.cmpf ogt, %in, %in_6 : f16
          %47 = arith.select %46, %in, %in_6 : f16
          linalg.yield %47 : f16
        } -> tensor<?x?x1xf16>
        %29 = tensor.empty(%0, %1) : tensor<?x?x32xf16>
        %30 = "loom.broadcast"(%28, %29) {dim = 2 : i64} : (tensor<?x?x1xf16>, tensor<?x?x32xf16>) -> tensor<?x?x?xf16>
        %31 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25 : tensor<?x?x?xf16>) outs(%23 : tensor<?x?x?xf16>) {
        ^bb0(%in: f16, %out: f16):
          %46 = arith.mulf %in, %cst_2 : f16
          linalg.yield %46 : f16
        } -> tensor<?x?x?xf16>
        %32 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%31, %30 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%23 : tensor<?x?x?xf16>) {
        ^bb0(%in: f16, %in_6: f16, %out: f16):
          %46 = arith.subf %in, %in_6 : f16
          linalg.yield %46 : f16
        } -> tensor<?x?x?xf16>
        %33 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%32 : tensor<?x?x?xf16>) outs(%23 : tensor<?x?x?xf16>) {
        ^bb0(%in: f16, %out: f16):
          %46 = math.exp %in : f16
          linalg.yield %46 : f16
        } -> tensor<?x?x?xf16>
        %34 = linalg.fill ins(%cst : f16) outs(%5 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
        %35 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%33 : tensor<?x?x?xf16>) outs(%34 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %46 = arith.addf %in, %out : f16
          linalg.yield %46 : f16
        } -> tensor<?x?x1xf16>
        %36 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg7, %28 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%5 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %in_6: f16, %out: f16):
          %46 = arith.subf %in, %in_6 : f16
          linalg.yield %46 : f16
        } -> tensor<?x?x1xf16>
        %37 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36 : tensor<?x?x1xf16>) outs(%5 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %46 = math.exp %in : f16
          linalg.yield %46 : f16
        } -> tensor<?x?x1xf16>
        %38 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg8, %37 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%5 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %in_6: f16, %out: f16):
          %46 = arith.mulf %in, %in_6 : f16
          linalg.yield %46 : f16
        } -> tensor<?x?x1xf16>
        %39 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38, %35 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%5 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %in_6: f16, %out: f16):
          %46 = arith.addf %in, %in_6 : f16
          linalg.yield %46 : f16
        } -> tensor<?x?x1xf16>
        %40 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %37 : tensor<?x?x128xf16>, tensor<?x?x1xf16>) outs(%8 : tensor<?x?x128xf16>) {
        ^bb0(%in: f16, %in_6: f16, %out: f16):
          %46 = arith.mulf %in, %in_6 : f16
          linalg.yield %46 : f16
        } -> tensor<?x?x128xf16>
        %subview_5 = memref.subview %v_view_arg[%10, %19, 0] [%0, %2, 128] [1, 1, 1] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
        %41 = bufferization.to_tensor %subview_5 : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to tensor<?x?x128xf16>
        cf.assert %22, "mismatching contracting dimension"
        %42 = arith.index_cast %2 : index to i64
        %43 = arith.cmpi eq, %42, %42 : i64
        cf.assert %43, "mismatching contracting dimension"
        %44 = linalg.batch_matmul ins(%33, %41 : tensor<?x?x?xf16>, tensor<?x?x128xf16>) outs(%9 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
        %45 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%44, %40 : tensor<?x?x128xf16>, tensor<?x?x128xf16>) outs(%8 : tensor<?x?x128xf16>) {
        ^bb0(%in: f16, %in_6: f16, %out: f16):
          %46 = arith.addf %in, %in_6 : f16
          linalg.yield %46 : f16
        } -> tensor<?x?x128xf16>
        scf.yield %28, %39, %45 : tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>
      }
      %15 = tensor.empty(%0, %1) : tensor<?x?x32xf16>
      %16 = "loom.broadcast"(%14#1, %15) {dim = 2 : i64} : (tensor<?x?x1xf16>, tensor<?x?x32xf16>) -> tensor<?x?x128xf16>
      %17 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%14#2, %16 : tensor<?x?x128xf16>, tensor<?x?x128xf16>) outs(%8 : tensor<?x?x128xf16>) {
      ^bb0(%in: f16, %in_4: f16, %out: f16):
        %19 = arith.divf %in, %in_4 : f16
        linalg.yield %19 : f16
      } -> tensor<?x?x128xf16>
      %subview_3 = memref.subview %out__arg[%10, %11, 0] [%0, %1, 128] [1, 1, 1] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
      %18 = bufferization.to_buffer %17 : tensor<?x?x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
      memref.copy %18, %subview_3 : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
    }
    return
  }
}


mlir-opt validation succeeded.

