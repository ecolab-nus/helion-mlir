module @helion_module {
  func.func private @torch.addmm(memref<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>)
  func.func @matmul(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>) -> memref<?x?xf32> {
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0 : memref<?x?xf32>
    %c1 = arith.constant 1 : index
    %dim_0 = memref.dim %arg1, %c1 : memref<?x?xf32>
    %alloc = memref.alloc(%dim, %dim_0) : memref<?x?xf32>
    %c0_1 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    scf.for %arg2 = %c0_1 to %dim step %c4 {
      %0 = arith.subi %dim, %arg2 : index
      %1 = arith.cmpi slt, %0, %c4 : index
      %2 = arith.select %1, %0, %c4 : index
      %c0_2 = arith.constant 0 : index
      %c4_3 = arith.constant 4 : index
      scf.for %arg3 = %c0_2 to %dim_0 step %c4_3 {
        %3 = arith.subi %dim_0, %arg3 : index
        %4 = arith.cmpi slt, %3, %c4_3 : index
        %5 = arith.select %4, %3, %c4_3 : index
        %alloc_4 = memref.alloc(%2, %5) : memref<?x?xf32>
        %cst = arith.constant 0.000000e+00 : f32
        linalg.fill ins(%cst : f32) outs(%alloc_4 : memref<?x?xf32>)
        %c0_5 = arith.constant 0 : index
        %c0_6 = arith.constant 0 : index
        %dim_7 = memref.dim %arg1, %c0_6 : memref<?x?xf32>
        %c8 = arith.constant 8 : index
        scf.for %arg4 = %c0_5 to %dim_7 step %c8 {
          %6 = arith.subi %dim_7, %arg4 : index
          %7 = arith.cmpi slt, %6, %c8 : index
          %8 = arith.select %7, %6, %c8 : index
          %subview_8 = memref.subview %arg0[%arg2, %arg4] [%2, %8] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
          %subview_9 = memref.subview %arg1[%arg4, %arg3] [%8, %5] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
          func.call @torch.addmm(%alloc_4, %subview_8, %subview_9) : (memref<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
        }
        %subview = memref.subview %alloc[%arg2, %arg3] [%2, %5] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        linalg.copy ins(%alloc_4 : memref<?x?xf32>) outs(%subview : memref<?x?xf32, strided<[?, 1], offset: ?>>)
      }
    }
    return %alloc : memref<?x?xf32>
  }
}

