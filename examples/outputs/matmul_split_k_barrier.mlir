#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.block_size_5 = -1 : index} {
  func.func @split_k_matmul(%arg0: memref<256x4096xf32>, %arg1: memref<4096x256xf32>, %arg2: memref<256x256x4096xf32>, %arg3: memref<256x256xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = "loom.get_symbol"() {name = "block_size_0"} : () -> index
    %1 = "loom.get_symbol"() {name = "block_size_1"} : () -> index
    %2 = "loom.get_symbol"() {name = "block_size_2"} : () -> index
    %3 = "loom.get_symbol"() {name = "block_size_5"} : () -> index
    affine.parallel (%arg4, %arg5, %arg6) = (0, 0, 0) to (256 ceildiv symbol(%0), 256 ceildiv symbol(%1), 4096 ceildiv symbol(%2)) {
      %4 = arith.muli %arg4, %0 : index
      %5 = arith.muli %arg6, %2 : index
      %subview = memref.subview %arg0[%4, %5] [%0, %2] [1, 1] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
      %6 = bufferization.to_tensor %subview : memref<?x?xf32, strided<[4096, 1], offset: ?>> to tensor<?x?xf32>
      %7 = arith.muli %arg5, %1 : index
      %subview_0 = memref.subview %arg1[%5, %7] [%2, %1] [1, 1] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
      %8 = bufferization.to_tensor %subview_0 : memref<?x?xf32, strided<[256, 1], offset: ?>> to tensor<?x?xf32>
      %9 = tensor.empty(%0, %1) : tensor<?x?xf32>
      %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %11 = linalg.matmul ins(%6, %8 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%10 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %subview_1 = memref.subview %arg2[%4, %7, %arg6] [%0, %1, 1] [1, 1, 1] : memref<256x256x4096xf32> to memref<?x?xf32, strided<[1048576, 4096], offset: ?>>
      %12 = bufferization.to_buffer %11 : tensor<?x?xf32> to memref<?x?xf32, strided<[1048576, 4096], offset: ?>>
      memref.copy %12, %subview_1 : memref<?x?xf32, strided<[1048576, 4096], offset: ?>> to memref<?x?xf32, strided<[1048576, 4096], offset: ?>>
    }
    affine.parallel (%arg4, %arg5) = (0, 0) to (256 ceildiv symbol(%0), 256 ceildiv symbol(%1)) {
      %4 = arith.muli %arg4, %0 : index
      %5 = arith.muli %arg5, %1 : index
      %subview = memref.subview %arg2[%4, %5, 0] [%0, %1, 4096] [1, 1, 1] : memref<256x256x4096xf32> to memref<?x?x4096xf32, strided<[1048576, 4096, 1], offset: ?>>
      %6 = bufferization.to_tensor %subview : memref<?x?x4096xf32, strided<[1048576, 4096, 1], offset: ?>> to tensor<?x?x4096xf32>
      %7 = tensor.empty(%0, %1) : tensor<?x?xf32>
      %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %9 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%6 : tensor<?x?x4096xf32>) outs(%8 : tensor<?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %11 = arith.addf %in, %out : f32
        linalg.yield %11 : f32
      } -> tensor<?x?xf32>
      %subview_0 = memref.subview %arg3[%4, %5] [%0, %1] [1, 1] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
      %10 = bufferization.to_buffer %9 : tensor<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
      memref.copy %10, %subview_0 : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32, strided<[256, 1], offset: ?>>
    }
    return
  }
}


