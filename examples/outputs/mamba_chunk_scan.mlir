=== Device IR ===
Graph 0: ForLoopGraphInfo
opcode         name                    target                                     args                                                                                  kwargs
-------------  ----------------------  -----------------------------------------  ------------------------------------------------------------------------------------  --------
placeholder    arg0_1                  arg0_1                                     ()                                                                                    {}
placeholder    arg1_1                  arg1_1                                     ()                                                                                    {}
call_function  _new_var                <function _new_var at 0x7f2fe4221870>      (arg0_1,)                                                                             {}
call_function  _new_var_1              <function _new_var at 0x7f2fe4221870>      (arg1_1,)                                                                             {}
call_function  block_size_4            <function _get_symnode at 0x7f2fe41fb2e0>  ('block_size_4',)                                                                     {}
call_function  tile_begin              <function tile_begin at 0x7f2fd56d7640>    (block_size_4,)                                                                       {}
call_function  block_size_3            <function _get_symnode at 0x7f2fe41fb2e0>  ('block_size_3',)                                                                     {}
call_function  tile_begin_1            <function tile_begin at 0x7f2fd56d7640>    (block_size_3,)                                                                       {}
call_function  block_size_5            <function _get_symnode at 0x7f2fe41fb2e0>  ('block_size_5',)                                                                     {}
call_function  tile_begin_2            <function tile_begin at 0x7f2fd56d7640>    (block_size_5,)                                                                       {}
call_function  x_size2                 <function _get_symnode at 0x7f2fe41fb2e0>  ('x_size2',)                                                                          {}
call_function  cb_size2                <function _get_symnode at 0x7f2fe41fb2e0>  ('cb_size2',)                                                                         {}
call_function  floordiv                <built-in function floordiv>               (x_size2, cb_size2)                                                                   {}
call_function  floordiv_1              <built-in function floordiv>               (tile_begin_2, floordiv)                                                              {}
call_function  cb                      <function _host_tensor at 0x7f2fe41fbf40>  ('cb',)                                                                               {}
call_function  sym_size_int            aten.sym_size.int                          (arg0_1, 0)                                                                           {}
call_function  block_size_2            <function _get_symnode at 0x7f2fe41fb2e0>  ('block_size_2',)                                                                     {}
call_function  cb_local                <function load at 0x7f2fd56c3ac0>          (cb, [tile_begin, tile_begin_1, floordiv_1, sym_size_int, block_size_2], None, None)  {}
call_function  dA_cumsum               <function _host_tensor at 0x7f2fe41fbf40>  ('dA_cumsum',)                                                                        {}
call_function  dA_cumsum_local_k       <function load at 0x7f2fd56c3ac0>          (dA_cumsum, [tile_begin, tile_begin_2, tile_begin_1, block_size_2], None, None)       {}
call_function  broadcast               <function broadcast at 0x7f2fd508da20>     (_new_var, 0, [block_size_2, sym_size_int])                                           {}
call_function  dA_cumsum_local_m_bc_k  aten.permute.default                       (broadcast, [1, 0])                                                                   {}
call_function  dA_cumsum_local_k_1     <function broadcast at 0x7f2fd508da20>     (dA_cumsum_local_k, 0, [sym_size_int, block_size_2])                                  {}
call_function  sub                     aten.sub.Tensor                            (dA_cumsum_local_m_bc_k, dA_cumsum_local_k_1)                                         {}
call_function  exp                     aten.exp.default                           (sub,)                                                                                {}
call_function  cb_local_1              aten.mul.Tensor                            (cb_local, exp)                                                                       {}
call_function  dt                      <function _host_tensor at 0x7f2fe41fbf40>  ('dt',)                                                                               {}
call_function  dt_local                <function load at 0x7f2fd56c3ac0>          (dt, [tile_begin, tile_begin_2, tile_begin_1, block_size_2], None, None)              {}
call_function  dt_local_1              <function broadcast at 0x7f2fd508da20>     (dt_local, 0, [sym_size_int, block_size_2])                                           {}
call_function  cb_local_2              aten.mul.Tensor                            (cb_local_1, dt_local_1)                                                              {}
call_function  cb_size3                <function _get_symnode at 0x7f2fe41fb2e0>  ('cb_size3',)                                                                         {}
call_function  mul_2                   <built-in function mul>                    (tile_begin_1, cb_size3)                                                              {}
call_function  tile_index              <function tile_index at 0x7f2fd56d72e0>    (block_size_2,)                                                                       {}
call_function  add                     aten.add.Tensor                            (tile_index, mul_2)                                                                   {}
call_function  x                       <function _host_tensor at 0x7f2fe41fbf40>  ('x',)                                                                                {}
call_function  sym_size_int_1          aten.sym_size.int                          (arg1_1, 1)                                                                           {}
call_function  x_local                 <function load at 0x7f2fd56c3ac0>          (x, [tile_begin, tile_begin_2, add, sym_size_int_1], None, None)                      {}
call_function  acc_o                   aten.addmm.default                         (_new_var_1, cb_local_2, x_local)                                                     {}
output         output                  output                                     ([acc_o],)                                                                            {}
Graph 1: ForLoopGraphInfo
opcode         name                    target                                     args                                                                                                          kwargs
-------------  ----------------------  -----------------------------------------  ------------------------------------------------------------------------------------------------------------  --------
call_function  block_size_0            <function _get_symnode at 0x7f2fe41fb2e0>  ('block_size_0',)                                                                                             {}
call_function  block_size_1            <function _get_symnode at 0x7f2fe41fb2e0>  ('block_size_1',)                                                                                             {}
call_function  acc_o                   <function full at 0x7f2fd56b72e0>          ([block_size_0, block_size_1], 0.0, torch.float16, None)                                                      {}
call_function  block_size_4            <function _get_symnode at 0x7f2fe41fb2e0>  ('block_size_4',)                                                                                             {}
call_function  tile_begin              <function tile_begin at 0x7f2fd56d7640>    (block_size_4,)                                                                                               {}
call_function  block_size_5            <function _get_symnode at 0x7f2fe41fb2e0>  ('block_size_5',)                                                                                             {}
call_function  tile_begin_1            <function tile_begin at 0x7f2fd56d7640>    (block_size_5,)                                                                                               {}
call_function  block_size_3            <function _get_symnode at 0x7f2fe41fb2e0>  ('block_size_3',)                                                                                             {}
call_function  tile_begin_2            <function tile_begin at 0x7f2fd56d7640>    (block_size_3,)                                                                                               {}
call_function  dA_cumsum               <function _host_tensor at 0x7f2fe41fbf40>  ('dA_cumsum',)                                                                                                {}
call_function  dA_cumsum_local_m       <function load at 0x7f2fd56c3ac0>          (dA_cumsum, [tile_begin, tile_begin_1, tile_begin_2, block_size_0], None, None)                               {}
call_function  broadcast               <function broadcast at 0x7f2fd508da20>     (dA_cumsum_local_m, 0, [block_size_1, block_size_0])                                                          {}
call_function  dA_cumsum_local_m_bc_n  aten.permute.default                       (broadcast, [1, 0])                                                                                           {}
call_function  scale_m_local           aten.exp.default                           (dA_cumsum_local_m_bc_n,)                                                                                     {}
call_function  x_size2                 <function _get_symnode at 0x7f2fe41fb2e0>  ('x_size2',)                                                                                                  {}
call_function  cb_size2                <function _get_symnode at 0x7f2fe41fb2e0>  ('cb_size2',)                                                                                                 {}
call_function  floordiv                <built-in function floordiv>               (x_size2, cb_size2)                                                                                           {}
call_function  floordiv_1              <built-in function floordiv>               (tile_begin_1, floordiv)                                                                                      {}
call_function  tile_index              <function tile_index at 0x7f2fd56d72e0>    (block_size_0,)                                                                                               {}
call_function  cb_size3                <function _get_symnode at 0x7f2fe41fb2e0>  ('cb_size3',)                                                                                                 {}
call_function  mul                     <built-in function mul>                    (tile_begin_2, cb_size3)                                                                                      {}
call_function  add                     aten.add.Tensor                            (tile_index, mul)                                                                                             {}
call_function  C                       <function _host_tensor at 0x7f2fe41fbf40>  ('C',)                                                                                                        {}
call_function  C_local                 <function load at 0x7f2fd56c3ac0>          (C, [tile_begin, floordiv_1, add, slice(None, None, None)], None, None)                                       {}
call_function  prev_states_T           <function _host_tensor at 0x7f2fe41fbf40>  ('prev_states_T',)                                                                                            {}
call_function  prev_states_local       <function load at 0x7f2fd56c3ac0>          (prev_states_T, [tile_begin, tile_begin_2, tile_begin_1, slice(None, None, None), block_size_1], None, None)  {}
call_function  acc_o_1                 <function dot at 0x7f2fe41f96c0>           (C_local, prev_states_local, acc_o, None)                                                                     {}
call_function  acc_o_2                 aten.mul.Tensor                            (acc_o_1, scale_m_local)                                                                                      {}
call_function  tile_id                 <function tile_id at 0x7f2fd56d7f40>       (block_size_0,)                                                                                               {}
call_function  add_1                   <built-in function add>                    (tile_id, 1)                                                                                                  {}
call_function  mul_2                   <built-in function mul>                    (add_1, block_size_0)                                                                                         {}
call_function  _for_loop               <function _for_loop at 0x7f2fe4220310>     (0, [0], [mul_2], [dA_cumsum_local_m, acc_o_2])                                                               {}
call_function  getitem                 <built-in function getitem>                (_for_loop, 0)                                                                                                {}
call_function  _phi                    <function _phi at 0x7f2fe4220820>          (acc_o_2, getitem)                                                                                            {}
call_function  D                       <function _host_tensor at 0x7f2fe41fbf40>  ('D',)                                                                                                        {}
call_function  D_local                 <function load at 0x7f2fd56c3ac0>          (D, [tile_begin_1], None, None)                                                                               {}
call_function  tile_index_1            <function tile_index at 0x7f2fd56d72e0>    (block_size_0,)                                                                                               {}
call_function  add_2                   aten.add.Tensor                            (tile_index_1, mul)                                                                                           {}
call_function  x                       <function _host_tensor at 0x7f2fe41fbf40>  ('x',)                                                                                                        {}
call_function  x_residual              <function load at 0x7f2fd56c3ac0>          (x, [tile_begin, tile_begin_1, add_2, block_size_1], None, None)                                              {}
call_function  mul_4                   aten.mul.Tensor                            (x_residual, D_local)                                                                                         {}
call_function  acc_o_3                 aten.add.Tensor                            (_phi, mul_4)                                                                                                 {}
call_function  tile_index_2            <function tile_index at 0x7f2fd56d72e0>    (block_size_0,)                                                                                               {}
call_function  add_4                   aten.add.Tensor                            (tile_index_2, mul)                                                                                           {}
call_function  out_                    <function _host_tensor at 0x7f2fe41fbf40>  ('out_',)                                                                                                     {}
call_function  store                   <function store at 0x7f2fd56c0160>         (out_, [tile_begin, tile_begin_1, add_4, block_size_1], acc_o_3, None)                                        {}
output         output                  output                                     ([],)                                                                                                         {}
Graph 2: ForLoopGraphInfo
opcode         name       target                                     args                     kwargs
-------------  ---------  -----------------------------------------  -----------------------  --------
call_function  x_size2    <function _get_symnode at 0x7f2fe41fb2e0>  ('x_size2',)             {}
call_function  _for_loop  <function _for_loop at 0x7f2fe4220310>     (1, [0], [x_size2], [])  {}
output         output     output                                     ([],)                    {}
Graph 3: RootGraphInfo
opcode         name       target                                     args                      kwargs
-------------  ---------  -----------------------------------------  ------------------------  --------
call_function  cb_size0   <function _get_symnode at 0x7f2fe41fb2e0>  ('cb_size0',)             {}
call_function  _for_loop  <function _for_loop at 0x7f2fe4220310>     (2, [0], [cb_size0], [])  {}
output         output     output                                     (None,)                   {}


=== Nodes with symbols ===
Node arg0_1 : FakeTensor(..., size=(u1,), dtype=torch.float16)
Node arg1_1 : FakeTensor(..., size=(u1, u2), dtype=torch.float16)
Node _new_var : FakeTensor(..., size=(u1,), dtype=torch.float16)
Node _new_var_1 : FakeTensor(..., size=(u1, u2), dtype=torch.float16)
Node block_size_4 : u12
Node tile_begin : u14
Node block_size_3 : u11
Node tile_begin_1 : u16
Node block_size_5 : u13
Node tile_begin_2 : u15
Node x_size2 : s53
Node cb_size2 : s77
Node floordiv : (s53//s77)
Node floordiv_1 : (u15//((s53//s77)))
Node cb : FakeTensor(..., size=(s64, s65, s77, s56, s50), dtype=torch.float16)
Node sym_size_int : u1
Node block_size_2 : u3
Node cb_local : FakeTensor(..., size=(u1, u3), dtype=torch.float16)
Node dA_cumsum : FakeTensor(..., size=(s12, s96, s42, s33), dtype=torch.float16)
Node dA_cumsum_local_k : FakeTensor(..., size=(u3,), dtype=torch.float16)
Node broadcast : FakeTensor(..., size=(u3, u1), dtype=torch.float16)
Node dA_cumsum_local_m_bc_k : FakeTensor(..., size=(u1, u3), dtype=torch.float16)
Node dA_cumsum_local_k_1 : FakeTensor(..., size=(u1, u3), dtype=torch.float16)
Node sub : FakeTensor(..., size=(u1, u3), dtype=torch.float16)
Node exp : FakeTensor(..., size=(u1, u3), dtype=torch.float16)
Node cb_local_1 : FakeTensor(..., size=(u1, u3), dtype=torch.float16)
Node dt : FakeTensor(..., size=(s41, s99, s89, s68), dtype=torch.float16)
Node dt_local : FakeTensor(..., size=(u3,), dtype=torch.float16)
Node dt_local_1 : FakeTensor(..., size=(u1, u3), dtype=torch.float16)
Node cb_local_2 : FakeTensor(..., size=(u1, u3), dtype=torch.float16)
Node cb_size3 : s56
Node mul_2 : s56*u16
Node tile_index : FakeTensor(..., size=(u3,), dtype=torch.int32)
Node add : FakeTensor(..., size=(u3,), dtype=torch.int32)
Node x : FakeTensor(..., size=(s78, s53, s27, s0), dtype=torch.float16)
Node sym_size_int_1 : u2
Node x_local : FakeTensor(..., size=(u3, u2), dtype=torch.float16)
Node acc_o : FakeTensor(..., size=(u1, u2), dtype=torch.float16)
Node block_size_0 : u1
Node block_size_1 : u2
Node acc_o : FakeTensor(..., size=(u1, u2), dtype=torch.float16)
Node block_size_4 : u12
Node tile_begin : u14
Node block_size_5 : u13
Node tile_begin_1 : u15
Node block_size_3 : u11
Node tile_begin_2 : u16
Node dA_cumsum : FakeTensor(..., size=(s12, s96, s42, s33), dtype=torch.float16)
Node dA_cumsum_local_m : FakeTensor(..., size=(u1,), dtype=torch.float16)
Node broadcast : FakeTensor(..., size=(u2, u1), dtype=torch.float16)
Node dA_cumsum_local_m_bc_n : FakeTensor(..., size=(u1, u2), dtype=torch.float16)
Node scale_m_local : FakeTensor(..., size=(u1, u2), dtype=torch.float16)
Node x_size2 : s53
Node cb_size2 : s77
Node floordiv : (s53//s77)
Node floordiv_1 : (u15//((s53//s77)))
Node tile_index : FakeTensor(..., size=(u1,), dtype=torch.int32)
Node cb_size3 : s56
Node mul : s56*u16
Node add : FakeTensor(..., size=(u1,), dtype=torch.int32)
Node C : FakeTensor(..., size=(s26, s5, s16, s22), dtype=torch.float16)
Node C_local : FakeTensor(..., size=(u1, u17), dtype=torch.float16)
Node prev_states_T : FakeTensor(..., size=(s92, s28, s61, s22, s46), dtype=torch.float16)
Node prev_states_local : FakeTensor(..., size=(u17, u2), dtype=torch.float16)
Node acc_o_1 : FakeTensor(..., size=(u1, u2), dtype=torch.float16)
Node acc_o_2 : FakeTensor(..., size=(u1, u2), dtype=torch.float16)
Node tile_id : u18
Node add_1 : u18 + 1
Node mul_2 : u1*(u18 + 1)
Node _for_loop : [FakeTensor(..., size=(u1, u2), dtype=torch.float16)]
Node getitem : FakeTensor(..., size=(u1, u2), dtype=torch.float16)
Node _phi : FakeTensor(..., size=(u1, u2), dtype=torch.float16)
Node D : FakeTensor(..., size=(s29,), dtype=torch.float16)
Node D_local : FakeTensor(..., size=(), dtype=torch.float16)
Node tile_index_1 : FakeTensor(..., size=(u1,), dtype=torch.int32)
Node add_2 : FakeTensor(..., size=(u1,), dtype=torch.int32)
Node x : FakeTensor(..., size=(s78, s53, s27, s0), dtype=torch.float16)
Node x_residual : FakeTensor(..., size=(u1, u2), dtype=torch.float16)
Node mul_4 : FakeTensor(..., size=(u1, u2), dtype=torch.float16)
Node acc_o_3 : FakeTensor(..., size=(u1, u2), dtype=torch.float16)
Node tile_index_2 : FakeTensor(..., size=(u1,), dtype=torch.int32)
Node add_4 : FakeTensor(..., size=(u1,), dtype=torch.int32)
Node out_ : FakeTensor(..., size=(s78, s53, s27, s0), dtype=torch.float16)
Node x_size2 : s53
Node _for_loop : []
Node cb_size0 : s64
Node _for_loop : []


=== Compile Environment ===
Block Sizes (7):
  Block 0: Size=s56, Var=u1, Reduction=False, Source=LoopSpecBlockSizeSource()
  Block 1: Size=s0, Var=u2, Reduction=False, Source=LoopSpecBlockSizeSource()
  Block 2: Size=u1*(u18 + 1), Var=u3, Reduction=False, Source=LoopSpecBlockSizeSource()
  Block 3: Size=s65, Var=u11, Reduction=False, Source=FixedBlockSizeSource(value=1)
  Block 4: Size=s64, Var=u12, Reduction=False, Source=LoopSpecBlockSizeSource()
  Block 5: Size=s53, Var=u13, Reduction=False, Source=LoopSpecBlockSizeSource()
  Block 6: Size=s22, Var=u17, Reduction=True, Source=ReductionLoopBlockSizeSource(reduction_loop=0)
Shape Env (38):
  Var s64: 2
  Var s65: 8
  Var s77: 4
  Var s56: 192
  Var s50: 192
  Var s78: 2
  Var s27: 1536
  Var s53: 32
  Var s0: 160
  Var s41: 2
  Var s99: 32
  Var s89: 8
  Var s68: 192
  Var s12: 2
  Var s96: 32
  Var s42: 8
  Var s33: 192
  Var s26: 2
  Var s16: 1536
  Var s5: 4
  Var s49: 128
  Var s92: 2
  Var s28: 8
  Var s61: 32
  Var s46: 160
  Var s22: 128
  Var s29: 32
  Var u1: 192
  Var u2: 160
  Var u3: 8192
  Var u11: 64
  Var u12: 64
  Var u13: 64
  Var u14: 8192
  Var u15: 8192
  Var u16: 8192
  Var u17: 128
  Var u18: 8192


=== MLIR Dump ===
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> ()>
module attributes {loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 32 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 192 : index}, loom.tile_n = {is_reduction = false, upper_bound = 160 : index}} {
  func.func @helion_mamba2_chunk_scan_kernel(%cb_arg: memref<2x8x4x192x192xf16>, %x_arg: memref<2x32x1536x160xf16>, %dt_arg: memref<2x32x8x192xf16>, %dA_cumsum_arg: memref<2x32x8x192xf16>, %C_arg: memref<2x4x1536x128xf16>, %D_arg: memref<32xf16>, %prev_states_T_arg: memref<2x8x32x128x160xf16>, %out__arg: memref<2x32x1536x160xf16>) {
    %cst = arith.constant 0.000000e+00 : f16
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c160 = arith.constant 160 : index
    %c192 = arith.constant 192 : index
    %c2 = arith.constant 2 : index
    %c32 = arith.constant 32 : index
    %0 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_m, upper_bound = 192 : index} : () -> index
    %1 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_n, upper_bound = 160 : index} : () -> index
    %2 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_k, upper_bound = 8192 : index} : () -> index
    %3 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_c, upper_bound = 8 : index} : () -> index
    %4 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_b, upper_bound = 2 : index} : () -> index
    %5 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_h, upper_bound = 32 : index} : () -> index
    %6 = arith.ceildivui %c192, %0 : index
    %7 = arith.ceildivui %c160, %1 : index
    %8 = arith.ceildivui %c8, %3 : index
    affine.parallel (%arg8, %arg9, %arg10) = (0, 0, 0) to (symbol(%6), symbol(%7), symbol(%8)) {
      %9 = arith.ceildivui %c2, %4 : index
      scf.for %arg11 = %c0 to %9 step %c1 {
        %10 = arith.ceildivui %c32, %5 : index
        scf.for %arg12 = %c0 to %10 step %c1 {
          %11 = tensor.empty(%0, %1) : tensor<?x?xf16>
          %12 = linalg.fill ins(%cst : f16) outs(%11 : tensor<?x?xf16>) -> tensor<?x?xf16>
          %13 = arith.muli %arg11, %4 : index
          %14 = arith.muli %arg12, %5 : index
          %15 = arith.muli %arg10, %3 : index
          %16 = arith.muli %arg8, %0 : index
          %subview = memref.subview %dA_cumsum_arg[%13, %14, %15, %16] [1, 1, 1, %0] [1, 1, 1, 1] : memref<2x32x8x192xf16> to memref<?xf16, strided<[1], offset: ?>>
          %17 = bufferization.to_tensor %subview : memref<?xf16, strided<[1], offset: ?>> to tensor<?xf16>
          %18 = tensor.empty(%1, %0) : tensor<?x?xf16>
          %19 = "loom.broadcast"(%17, %18) {dim = 0 : i64} : (tensor<?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
          %transposed = linalg.transpose ins(%19 : tensor<?x?xf16>) outs(%11 : tensor<?x?xf16>) permutation = [1, 0] 
          %20 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%transposed : tensor<?x?xf16>) outs(%11 : tensor<?x?xf16>) {
          ^bb0(%in: f16, %out: f16):
            %38 = math.exp %in : f16
            linalg.yield %38 : f16
          } -> tensor<?x?xf16>
          %21 = arith.divui %14, %c8 : index
          %22 = arith.muli %15, %c192 : index
          %23 = arith.addi %16, %22 : index
          %subview_0 = memref.subview %C_arg[%13, %21, %23, 0] [1, 1, %0, 128] [1, 1, 1, 1] : memref<2x4x1536x128xf16> to memref<?x128xf16, strided<[128, 1], offset: ?>>
          %24 = bufferization.to_tensor %subview_0 : memref<?x128xf16, strided<[128, 1], offset: ?>> to tensor<?x128xf16>
          %25 = arith.muli %arg9, %1 : index
          %subview_1 = memref.subview %prev_states_T_arg[%13, %15, %14, 0, %25] [1, 1, 1, 128, %1] [1, 1, 1, 1, 1] : memref<2x8x32x128x160xf16> to memref<128x?xf16, strided<[160, 1], offset: ?>>
          %26 = bufferization.to_tensor %subview_1 : memref<128x?xf16, strided<[160, 1], offset: ?>> to tensor<128x?xf16>
          %27 = linalg.matmul ins(%24, %26 : tensor<?x128xf16>, tensor<128x?xf16>) outs(%12 : tensor<?x?xf16>) -> tensor<?x?xf16>
          %28 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%27, %20 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%11 : tensor<?x?xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %38 = arith.mulf %in, %in_5 : f16
            linalg.yield %38 : f16
          } -> tensor<?x?xf16>
          %29 = arith.addi %arg8, %c1 : index
          %30 = arith.muli %29, %0 : index
          %31 = arith.ceildivui %30, %2 : index
          %32 = scf.for %arg13 = %c0 to %31 step %c1 iter_args(%arg14 = %28) -> (tensor<?x?xf16>) {
            %38 = arith.muli %arg13, %2 : index
            %subview_5 = memref.subview %cb_arg[%13, %15, %21, %16, %38] [1, 1, 1, %0, %2] [1, 1, 1, 1, 1] : memref<2x8x4x192x192xf16> to memref<?x?xf16, strided<[192, 1], offset: ?>>
            %39 = bufferization.to_tensor %subview_5 : memref<?x?xf16, strided<[192, 1], offset: ?>> to tensor<?x?xf16>
            %subview_6 = memref.subview %dA_cumsum_arg[%13, %14, %15, %38] [1, 1, 1, %2] [1, 1, 1, 1] : memref<2x32x8x192xf16> to memref<?xf16, strided<[1], offset: ?>>
            %40 = bufferization.to_tensor %subview_6 : memref<?xf16, strided<[1], offset: ?>> to tensor<?xf16>
            %41 = tensor.empty(%2, %0) : tensor<?x?xf16>
            %42 = "loom.broadcast"(%17, %41) {dim = 0 : i64} : (tensor<?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
            %43 = tensor.empty(%0, %2) : tensor<?x?xf16>
            %transposed_7 = linalg.transpose ins(%42 : tensor<?x?xf16>) outs(%43 : tensor<?x?xf16>) permutation = [1, 0] 
            %44 = "loom.broadcast"(%40, %43) {dim = 0 : i64} : (tensor<?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
            %45 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%transposed_7, %44 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%43 : tensor<?x?xf16>) {
            ^bb0(%in: f16, %in_10: f16, %out: f16):
              %55 = arith.subf %in, %in_10 : f16
              linalg.yield %55 : f16
            } -> tensor<?x?xf16>
            %46 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%45 : tensor<?x?xf16>) outs(%43 : tensor<?x?xf16>) {
            ^bb0(%in: f16, %out: f16):
              %55 = math.exp %in : f16
              linalg.yield %55 : f16
            } -> tensor<?x?xf16>
            %47 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%39, %46 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%43 : tensor<?x?xf16>) {
            ^bb0(%in: f16, %in_10: f16, %out: f16):
              %55 = arith.mulf %in, %in_10 : f16
              linalg.yield %55 : f16
            } -> tensor<?x?xf16>
            %subview_8 = memref.subview %dt_arg[%13, %14, %15, %38] [1, 1, 1, %2] [1, 1, 1, 1] : memref<2x32x8x192xf16> to memref<?xf16, strided<[1], offset: ?>>
            %48 = bufferization.to_tensor %subview_8 : memref<?xf16, strided<[1], offset: ?>> to tensor<?xf16>
            %49 = "loom.broadcast"(%48, %43) {dim = 0 : i64} : (tensor<?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
            %50 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%47, %49 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%43 : tensor<?x?xf16>) {
            ^bb0(%in: f16, %in_10: f16, %out: f16):
              %55 = arith.mulf %in, %in_10 : f16
              linalg.yield %55 : f16
            } -> tensor<?x?xf16>
            %51 = arith.addi %38, %22 : index
            %subview_9 = memref.subview %x_arg[%13, %14, %51, %25] [1, 1, %2, %1] [1, 1, 1, 1] : memref<2x32x1536x160xf16> to memref<?x?xf16, strided<[160, 1], offset: ?>>
            %52 = bufferization.to_tensor %subview_9 : memref<?x?xf16, strided<[160, 1], offset: ?>> to tensor<?x?xf16>
            %53 = linalg.matmul ins(%50, %52 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%12 : tensor<?x?xf16>) -> tensor<?x?xf16>
            %54 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg14, %53 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%11 : tensor<?x?xf16>) {
            ^bb0(%in: f16, %in_10: f16, %out: f16):
              %55 = arith.addf %in, %in_10 : f16
              linalg.yield %55 : f16
            } -> tensor<?x?xf16>
            scf.yield %54 : tensor<?x?xf16>
          }
          %subview_2 = memref.subview %D_arg[%14] [1] [1] : memref<32xf16> to memref<f16, strided<[], offset: ?>>
          %33 = bufferization.to_tensor %subview_2 : memref<f16, strided<[], offset: ?>> to tensor<f16>
          %subview_3 = memref.subview %x_arg[%13, %14, %23, %25] [1, 1, %0, %1] [1, 1, 1, 1] : memref<2x32x1536x160xf16> to memref<?x?xf16, strided<[160, 1], offset: ?>>
          %34 = bufferization.to_tensor %subview_3 : memref<?x?xf16, strided<[160, 1], offset: ?>> to tensor<?x?xf16>
          %35 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%34, %33 : tensor<?x?xf16>, tensor<f16>) outs(%11 : tensor<?x?xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %38 = arith.mulf %in, %in_5 : f16
            linalg.yield %38 : f16
          } -> tensor<?x?xf16>
          %36 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%32, %35 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%11 : tensor<?x?xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %38 = arith.addf %in, %in_5 : f16
            linalg.yield %38 : f16
          } -> tensor<?x?xf16>
          %subview_4 = memref.subview %out__arg[%13, %14, %23, %25] [1, 1, %0, %1] [1, 1, 1, 1] : memref<2x32x1536x160xf16> to memref<?x?xf16, strided<[160, 1], offset: ?>>
          %37 = bufferization.to_buffer %36 : tensor<?x?xf16> to memref<?x?xf16, strided<[160, 1], offset: ?>>
          memref.copy %37, %subview_4 : memref<?x?xf16, strided<[160, 1], offset: ?>> to memref<?x?xf16, strided<[160, 1], offset: ?>>
        }
      }
    }
    return
  }
}


mlir-opt validation succeeded.

