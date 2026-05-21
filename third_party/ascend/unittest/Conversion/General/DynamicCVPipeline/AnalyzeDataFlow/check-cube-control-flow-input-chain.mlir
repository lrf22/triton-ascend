#map = affine_map<(d0) -> (d0)>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_cube_control_flow_input_chain(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg10: memref<?xf32> {tt.divisibility = 16 : i32}, %arg11: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg12: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg13: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i64 {tt.divisibility = 16 : i32}, %arg16: i64 {tt.divisibility = 16 : i32}, %arg17: i64 {tt.divisibility = 16 : i32}, %arg18: i64 {tt.divisibility = 16 : i32}, %arg19: i64 {tt.divisibility = 16 : i32}, %arg20: i64 {tt.divisibility = 16 : i32}, %arg21: i64 {tt.divisibility = 16 : i32}, %arg22: i64 {tt.divisibility = 16 : i32}, %arg23: i64 {tt.divisibility = 16 : i32}, %arg24: i64 {tt.divisibility = 16 : i32}, %arg25: i64 {tt.divisibility = 16 : i32}, %arg26: i64 {tt.divisibility = 16 : i32}, %arg27: i64 {tt.divisibility = 16 : i32}, %arg28: i64 {tt.divisibility = 16 : i32}, %arg29: i64 {tt.divisibility = 16 : i32}, %arg30: i64 {tt.divisibility = 16 : i32}, %arg31: i64 {tt.divisibility = 16 : i32}, %arg32: i64 {tt.divisibility = 16 : i32}, %arg33: i64 {tt.divisibility = 16 : i32}, %arg34: i64 {tt.divisibility = 16 : i32}, %arg35: i64 {tt.divisibility = 16 : i32}, %arg36: i32, %arg37: i32, %arg38: i32, %arg39: i32, %arg40: i32, %arg41: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c128_i32 = arith.constant {ssbuffer.block_id = 22 : i32} 128 : i32
    %cst = arith.constant {ssbuffer.block_id = 22 : i32} 1.000000e+00 : f32
    %cst_0 = arith.constant {ssbuffer.block_id = 15 : i32} dense<[4, 4, 16, 8]> : tensor<4xi64>
    %cst_1 = arith.constant {ssbuffer.block_id = 15 : i32} dense<[64, 4, 8]> : tensor<3xi64>
    %cst_2 = arith.constant {ssbuffer.block_id = 20 : i32} 0.000000e+00 : f32
    %c1_i64 = arith.constant {ssbuffer.block_id = 20 : i32} 1 : i64
    %c64_i32 = arith.constant {MixUse, ssbuffer.block_id = 20 : i32} 64 : i32
    %c1_i32 = arith.constant {ssbuffer.block_id = 20 : i32} 1 : i32
    %c4_i32 = arith.constant {ssbuffer.block_id = 20 : i32} 4 : i32
    %c2_i32 = arith.constant {MixUse, ssbuffer.block_id = 20 : i32} 2 : i32
    %c0_i64 = arith.constant {Undefined, ssbuffer.block_id = 20 : i32} 0 : i64
    %c32_i64 = arith.constant {ssbuffer.block_id = 20 : i32} 32 : i64
    %c1 = arith.constant {ssbuffer.block_id = 20 : i32} 1 : index
    %c-1 = arith.constant {ssbuffer.block_id = 20 : i32} -1 : index
    %c-1_i64 = arith.constant {ssbuffer.block_id = 20 : i32} -1 : i64
    %c64 = arith.constant {ssbuffer.block_id = 20 : i32} 64 : index
    %c32 = arith.constant {ssbuffer.block_id = 20 : i32} 32 : index
    %c128 = arith.constant {ssbuffer.block_id = 20 : i32} 128 : index
    %c0 = arith.constant {ssbuffer.block_id = 6 : i32} 0 : index
    scope.scope : () -> () {
      %0 = tensor.empty() {ssbuffer.block_id = 20 : i32} : tensor<64x32xf32>
      %1 = linalg.fill {ssbuffer.block_id = 20 : i32} ins(%cst_2 : f32) outs(%0 : tensor<64x32xf32>) -> tensor<64x32xf32>
      %2 = arith.extsi %arg40 {ssbuffer.block_id = 20 : i32} : i32 to i64
      %3 = arith.divsi %arg39, %c2_i32 {MixUse, ssbuffer.block_id = 20 : i32} : i32
      %4 = arith.remsi %arg39, %c2_i32 {ssbuffer.block_id = 20 : i32} : i32
      %5 = arith.muli %2, %arg15 {ssbuffer.block_id = 20 : i32} : i64
      %6 = arith.divsi %arg41, %c4_i32 {ssbuffer.block_id = 20 : i32} : i32
      %7 = arith.extsi %6 {ssbuffer.block_id = 20 : i32} : i32 to i64
      %8 = arith.muli %7, %arg16 {ssbuffer.block_id = 20 : i32} : i64
      %9 = arith.addi %5, %8 {ssbuffer.block_id = 20 : i32} : i64
      %10 = arith.index_cast %9 {ssbuffer.block_id = 20 : i32} : i64 to index
      %11 = arith.index_cast %arg40 {ssbuffer.block_id = 20 : i32} : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg13 to offset: [%11], sizes: [1], strides: [1] {ssbuffer.block_id = 20 : i32} : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
      %12 = memref.load %reinterpret_cast[%c0] {ssbuffer.block_id = 20 : i32} : memref<1xi64, strided<[1], offset: ?>>
      %13 = arith.addi %11, %c1 {ssbuffer.block_id = 20 : i32} : index
      %reinterpret_cast_3 = memref.reinterpret_cast %arg13 to offset: [%13], sizes: [1], strides: [1] {ssbuffer.block_id = 20 : i32} : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
      %14 = memref.load %reinterpret_cast_3[%c0] {ssbuffer.block_id = 20 : i32} : memref<1xi64, strided<[1], offset: ?>>
      %15 = arith.extsi %arg41 {ssbuffer.block_id = 20 : i32} : i32 to i64
      %16 = arith.muli %2, %arg24 {ssbuffer.block_id = 20 : i32} : i64
      %17 = arith.muli %15, %arg25 {ssbuffer.block_id = 20 : i32} : i64
      %18 = arith.addi %16, %17 {ssbuffer.block_id = 20 : i32} : i64
      %19 = arith.index_cast %18 {ssbuffer.block_id = 20 : i32} : i64 to index
      %20 = arith.muli %2, %arg26 {ssbuffer.block_id = 20 : i32} : i64
      %21 = arith.muli %15, %arg27 {ssbuffer.block_id = 20 : i32} : i64
      %22 = arith.addi %20, %21 {ssbuffer.block_id = 20 : i32} : i64
      %23 = arith.index_cast %22 {ssbuffer.block_id = 20 : i32} : i64 to index
      %24 = arith.muli %3, %c64_i32 {MixUse, ssbuffer.block_id = 20 : i32} : i32
      %25 = tensor.empty() {ssbuffer.block_id = 20 : i32} : tensor<64xi32>
      %26 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%25 : tensor<64xi32>) attrs =  {ssbuffer.block_id = 20 : i32, tt.from_make_range, tt.make_range_offset = 0 : index, tt.make_range_size = 64 : index} {
      ^bb0(%out: i32):
        %111 = linalg.index 0 : index
        %112 = arith.index_cast %111 : index to i32
        linalg.yield %112 : i32
      } -> tensor<64xi32>
      %27 = linalg.fill {ssbuffer.block_id = 20 : i32} ins(%24 : i32) outs(%25 : tensor<64xi32>) -> tensor<64xi32>
      %28 = arith.addi %27, %26 {MixUse, ssbuffer.block_id = 20 : i32} : tensor<64xi32>
      %29 = arith.subi %14, %12 {ssbuffer.block_id = 20 : i32} : i64
      %30 = arith.muli %4, %c64_i32 {ssbuffer.block_id = 20 : i32} : i32
      %31 = arith.index_cast %30 {ssbuffer.block_id = 20 : i32} : i32 to index
      %32 = arith.index_cast %24 {ssbuffer.block_id = 20 : i32} : i32 to index
      %33 = arith.addi %32, %c64 {ssbuffer.block_id = 20 : i32} : index
      %34 = arith.maxsi %32, %c32 {ssbuffer.block_id = 20 : i32} : index
      %35 = arith.minsi %33, %34 {ssbuffer.block_id = 20 : i32} : index
      %36 = arith.subi %35, %32 {ssbuffer.block_id = 20 : i32} : index
      %37 = arith.cmpi slt, %36, %c64 {ssbuffer.block_id = 20 : i32} : index
      %expanded = tensor.expand_shape %28 [[0, 1]] output_shape [64, 1] {ssbuffer.block_id = 20 : i32} : tensor<64xi32> into tensor<64x1xi32>
      %38 = arith.extsi %expanded {MixUse, ssbuffer.block_id = 20 : i32} : tensor<64x1xi32> to tensor<64x1xi64>
      %39 = arith.index_cast %29 {ssbuffer.block_id = 20 : i32} : i64 to index
      %40 = arith.maxsi %32, %39 {ssbuffer.block_id = 20 : i32} : index
      %41 = arith.minsi %33, %40 {ssbuffer.block_id = 20 : i32} : index
      %42 = arith.subi %41, %32 {ssbuffer.block_id = 20 : i32} : index
      %43 = arith.minsi %42, %c64 {ssbuffer.block_id = 20 : i32} : index
      %44 = arith.cmpi slt, %43, %c64 {ssbuffer.block_id = 20 : i32} : index
      %45 = arith.addi %31, %c64 {ssbuffer.block_id = 20 : i32} : index
      %46 = arith.maxsi %31, %c128 {ssbuffer.block_id = 20 : i32} : index
      %47 = arith.minsi %45, %46 {ssbuffer.block_id = 20 : i32} : index
      %48 = arith.subi %47, %31 {ssbuffer.block_id = 20 : i32} : index
      %49 = arith.minsi %48, %c64 {ssbuffer.block_id = 20 : i32} : index
      %50 = arith.cmpi slt, %49, %c64 {ssbuffer.block_id = 20 : i32} : index
      %51 = tensor.empty() {ssbuffer.block_id = 20 : i32} : tensor<32xi32>
      %52 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%51 : tensor<32xi32>) attrs =  {ssbuffer.block_id = 20 : i32, tt.from_make_range, tt.make_range_offset = 0 : index, tt.make_range_size = 32 : index} {
      ^bb0(%out: i32):
        %111 = linalg.index 0 : index
        %112 = arith.index_cast %111 : index to i32
        linalg.yield %112 : i32
      } -> tensor<32xi32>
      %expanded_4 = tensor.expand_shape %52 [[0, 1]] output_shape [1, 32] {ssbuffer.block_id = 20 : i32} : tensor<32xi32> into tensor<1x32xi32>
      %53 = arith.extsi %expanded_4 {MixUse, ssbuffer.block_id = 20 : i32} : tensor<1x32xi32> to tensor<1x32xi64>
      %54 = arith.addi %3, %c1_i32 {Undefined, ssbuffer.block_id = 20 : i32} : i32
      %55 = arith.muli %54, %c64_i32 {Undefined, ssbuffer.block_id = 20 : i32} : i32
      %56 = arith.extsi %55 {Undefined, ssbuffer.block_id = 20 : i32} : i32 to i64
      %57 = arith.minsi %56, %29 {Undefined, ssbuffer.block_id = 20 : i32} : i64
      %58 = tensor.empty() {ssbuffer.block_id = 20 : i32} : tensor<64x32xi64>
      %collapsed = tensor.collapse_shape %38 [[0, 1]] {ssbuffer.block_id = 20 : i32} : tensor<64x1xi64> into tensor<64xi64>
      %broadcasted = linalg.broadcast ins(%collapsed : tensor<64xi64>) outs(%58 : tensor<64x32xi64>) dimensions = [1]  {ssbuffer.block_id = 20 : i32}
      %59 = arith.index_cast %arg17 {ssbuffer.block_id = 20 : i32} : i64 to index
      %60 = arith.muli %32, %59 {ssbuffer.block_id = 20 : i32} : index
      %61 = arith.cmpi slt, %48, %c64 {ssbuffer.block_id = 20 : i32} : index
      %62 = arith.ori %44, %50 {ssbuffer.block_id = 20 : i32} : i1
      hivm.hir.sync_block_wait {ssbuffer.block_id = 21 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 3
      %alloc = memref.alloc() {ssbuffer.block_id = 21 : i32, ssbuffer.transfer_id = 2 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 21 : i32, ssbuffer.transfer_id = 2 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      %memspacecast = memref.memory_space_cast %alloc {ssbuffer.block_id = 21 : i32, ssbuffer.transfer_id = 2 : i32} : memref<64x64xf32, #hivm.address_space<ub>> to memref<64x64xf32>
      %63 = bufferization.to_tensor %memspacecast restrict writable {ssbuffer.block_id = 21 : i32, ssbuffer.transfer_id = 2 : i32} : memref<64x64xf32>
      %alloc_5 = memref.alloc() {ssbuffer.block_id = 21 : i32} : memref<64xf32>
      scf.if %37 {
        linalg.fill {ssbuffer.block_id = 21 : i32} ins(%cst_2 : f32) outs(%alloc_5 : memref<64xf32>)
      } {hivm.unlikely_condition, ssbuffer.block_id = 21 : i32}
      %64 = tensor.empty() {ssbuffer.block_id = 21 : i32} : tensor<64x64xf32>
      %65 = linalg.fill {ssbuffer.block_id = 21 : i32} ins(%cst_2 : f32) outs(%64 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %66 = arith.addi %23, %32 {ssbuffer.block_id = 21 : i32} : index
      %reinterpret_cast_6 = memref.reinterpret_cast %arg7 to offset: [%66], sizes: [64], strides: [1] {ssbuffer.block_id = 21 : i32} : memref<?xf32> to memref<64xf32, strided<[1], offset: ?>>
      %subview = memref.subview %reinterpret_cast_6[0] [%36] [1] {ssbuffer.block_id = 21 : i32} : memref<64xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
      %subview_7 = memref.subview %alloc_5[0] [%36] [1] {ssbuffer.block_id = 21 : i32} : memref<64xf32> to memref<?xf32, strided<[1]>>
      memref.copy %subview, %subview_7 {ssbuffer.block_id = 21 : i32} : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
      %67 = bufferization.to_tensor %alloc_5 restrict writable {ssbuffer.block_id = 21 : i32} : memref<64xf32>
      %68 = math.exp %67 {DataUse, ssbuffer.block_id = 21 : i32} : tensor<64xf32>
      %69 = arith.addf %63, %65 {ssbuffer.block_id = 21 : i32} : tensor<64x64xf32>
      %broadcasted_8 = linalg.broadcast ins(%68 : tensor<64xf32>) outs(%64 : tensor<64x64xf32>) dimensions = [1]  {ssbuffer.block_id = 21 : i32}
      %70 = arith.mulf %69, %broadcasted_8 {DataUse, ssbuffer.block_id = 21 : i32} : tensor<64x64xf32>
      %broadcasted_9 = linalg.broadcast ins(%67 : tensor<64xf32>) outs(%0 : tensor<64x32xf32>) dimensions = [1]  {ssbuffer.block_id = 21 : i32}
      %alloc_10 = memref.alloc() {ssbuffer.block_id = 24 : i32, ssbuffer.transfer_id = 0 : i32} : memref<4x4x16x8xf32, #hivm.address_space<cbuf>>
      annotation.mark %alloc_10 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 24 : i32, ssbuffer.transfer_id = 0 : i32} : memref<4x4x16x8xf32, #hivm.address_space<cbuf>>
      %alloc_11 = memref.alloc() {ssbuffer.block_id = 24 : i32, ssbuffer.transfer_id = 1 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_11 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 24 : i32, ssbuffer.transfer_id = 1 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      hivm.hir.sync_block_set {ssbuffer.block_id = 24 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 2
      %71:4 = scf.for %arg42 = %c0_i64 to %57 step %c32_i64 iter_args(%arg43 = %70, %arg44 = %60, %arg45 = %c0, %arg46 = %c0) -> (tensor<64x64xf32>, index, index, index)  : i64 {
        %111 = arith.subi %c32_i64, %arg42 {ssbuffer.block_id = 14 : i32} : i64
        %112 = arith.index_cast %111 {ssbuffer.block_id = 14 : i32} : i64 to index
        %113 = arith.maxsi %112, %c0 {ssbuffer.block_id = 14 : i32} : index
        %114 = arith.minsi %113, %c32 {ssbuffer.block_id = 14 : i32} : index
        %115 = arith.minsi %36, %c64 {ssbuffer.block_id = 14 : i32} : index
        %116 = arith.minsi %114, %c32 {ssbuffer.block_id = 14 : i32} : index
        %117 = arith.cmpi slt, %115, %c64 {ssbuffer.block_id = 14 : i32} : index
        %118 = arith.cmpi slt, %116, %c32 {ssbuffer.block_id = 14 : i32} : index
        %119 = arith.ori %117, %118 {ssbuffer.block_id = 14 : i32} : i1
        %120 = arith.cmpi slt, %114, %c32 {ssbuffer.block_id = 14 : i32} : index
        %121 = arith.addi %10, %arg44 {ssbuffer.block_id = 14 : i32} : index
        %reinterpret_cast_27 = memref.reinterpret_cast %arg2 to offset: [%121], sizes: [64, 32], strides: [%59, %c1] {ssbuffer.block_id = 14 : i32} : memref<?xf32> to memref<64x32xf32, strided<[?, ?], offset: ?>>
        %122 = arith.addi %19, %arg45 {ssbuffer.block_id = 14 : i32} : index
        %reinterpret_cast_28 = memref.reinterpret_cast %arg6 to offset: [%122], sizes: [32], strides: [%c1] {ssbuffer.block_id = 14 : i32} : memref<?xf32> to memref<32xf32, strided<[?], offset: ?>>
        %123 = arith.addi %23, %arg46 {ssbuffer.block_id = 14 : i32} : index
        %reinterpret_cast_29 = memref.reinterpret_cast %arg7 to offset: [%123], sizes: [32], strides: [%c1] {ssbuffer.block_id = 14 : i32} : memref<?xf32> to memref<32xf32, strided<[?], offset: ?>>
        %subview_30 = memref.subview %reinterpret_cast_27[0, 0] [%115, %116] [1, 1] {ssbuffer.block_id = 14 : i32} : memref<64x32xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %subview_31 = memref.subview %reinterpret_cast_29[0] [%114] [1] {ssbuffer.block_id = 14 : i32} : memref<32xf32, strided<[?], offset: ?>> to memref<?xf32, strided<[?], offset: ?>>
        %subview_32 = memref.subview %reinterpret_cast_28[0] [%114] [1] {ssbuffer.block_id = 14 : i32} : memref<32xf32, strided<[?], offset: ?>> to memref<?xf32, strided<[?], offset: ?>>
        %124 = arith.addi %arg44, %c32 {ssbuffer.block_id = 14 : i32} : index
        %125 = arith.addi %arg45, %c32 {ssbuffer.block_id = 14 : i32} : index
        %126 = arith.addi %arg46, %c32 {ssbuffer.block_id = 14 : i32} : index
        %alloc_33 = memref.alloc() {ssbuffer.block_id = 15 : i32} : memref<64x32xf32>
        %alloc_34 = memref.alloc() {ssbuffer.block_id = 15 : i32} : memref<32xf32>
        %alloc_35 = memref.alloc() {ssbuffer.block_id = 15 : i32} : memref<32xf32>
        %subview_36 = memref.subview %alloc_33[0, 0] [%115, %116] [1, 1] {ssbuffer.block_id = 15 : i32} : memref<64x32xf32> to memref<?x?xf32, strided<[32, 1]>>
        %subview_37 = memref.subview %alloc_34[0] [%114] [1] {ssbuffer.block_id = 15 : i32} : memref<32xf32> to memref<?xf32, strided<[1]>>
        %subview_38 = memref.subview %alloc_35[0] [%114] [1] {ssbuffer.block_id = 15 : i32} : memref<32xf32> to memref<?xf32, strided<[1]>>
        scf.if %120 {
          linalg.fill {ssbuffer.block_id = 15 : i32} ins(%cst_2 : f32) outs(%alloc_35 : memref<32xf32>)
          linalg.fill {ssbuffer.block_id = 15 : i32} ins(%cst_2 : f32) outs(%alloc_34 : memref<32xf32>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 15 : i32}
        scf.if %119 {
          linalg.fill {ssbuffer.block_id = 15 : i32} ins(%cst_2 : f32) outs(%alloc_33 : memref<64x32xf32>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 15 : i32}
        memref.copy %subview_30, %subview_36 {ssbuffer.block_id = 15 : i32} : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[32, 1]>>
        %127 = bufferization.to_tensor %alloc_33 restrict writable {ssbuffer.block_id = 15 : i32} : memref<64x32xf32>
        memref.copy %subview_31, %subview_37 {ssbuffer.block_id = 15 : i32} : memref<?xf32, strided<[?], offset: ?>> to memref<?xf32, strided<[1]>>
        %128 = bufferization.to_tensor %alloc_34 restrict writable {ssbuffer.block_id = 15 : i32} : memref<32xf32>
        %broadcasted_39 = linalg.broadcast ins(%128 : tensor<32xf32>) outs(%0 : tensor<64x32xf32>) dimensions = [0]  {ssbuffer.block_id = 15 : i32}
        %129 = arith.subf %broadcasted_9, %broadcasted_39 {DataUse, ssbuffer.block_id = 15 : i32} : tensor<64x32xf32>
        %130 = math.exp %129 {DataUse, ssbuffer.block_id = 15 : i32} : tensor<64x32xf32>
        %131 = arith.mulf %127, %130 {DataUse, ssbuffer.block_id = 15 : i32} : tensor<64x32xf32>
        memref.copy %subview_32, %subview_38 {ssbuffer.block_id = 15 : i32} : memref<?xf32, strided<[?], offset: ?>> to memref<?xf32, strided<[1]>>
        %132 = bufferization.to_tensor %alloc_35 restrict writable {ssbuffer.block_id = 15 : i32} : memref<32xf32>
        %broadcasted_40 = linalg.broadcast ins(%132 : tensor<32xf32>) outs(%0 : tensor<64x32xf32>) dimensions = [0]  {ssbuffer.block_id = 15 : i32}
        %133 = arith.mulf %131, %broadcasted_40 {DataUse, ssbuffer.block_id = 15 : i32} : tensor<64x32xf32>
        %134 = tensor.empty() {ssbuffer.block_id = 15 : i32} : tensor<1x32xi64>
        %135 = linalg.fill {ssbuffer.block_id = 15 : i32} ins(%arg42 : i64) outs(%134 : tensor<1x32xi64>) -> tensor<1x32xi64>
        %136 = arith.addi %135, %53 {DataUse, ssbuffer.block_id = 15 : i32} : tensor<1x32xi64>
        %collapsed_41 = tensor.collapse_shape %136 [[0, 1]] {ssbuffer.block_id = 15 : i32} : tensor<1x32xi64> into tensor<32xi64>
        %broadcasted_42 = linalg.broadcast ins(%collapsed_41 : tensor<32xi64>) outs(%58 : tensor<64x32xi64>) dimensions = [0]  {ssbuffer.block_id = 15 : i32}
        %137 = arith.cmpi sge, %broadcasted, %broadcasted_42 {DataUse, ssbuffer.block_id = 15 : i32} : tensor<64x32xi64>
        %138 = arith.select %137, %133, %1 {DataUse, ssbuffer.block_id = 15 : i32} : tensor<64x32xi1>, tensor<64x32xf32>
        %reshape = tensor.reshape %138(%cst_1) {ssbuffer.block_id = 15 : i32} : (tensor<64x32xf32>, tensor<3xi64>) -> tensor<64x4x8xf32>
        %139 = tensor.empty() {ssbuffer.block_id = 15 : i32} : tensor<4x64x8xf32>
        %transposed = linalg.transpose ins(%reshape : tensor<64x4x8xf32>) outs(%139 : tensor<4x64x8xf32>) permutation = [1, 0, 2]  {ssbuffer.block_id = 15 : i32}
        %reshape_43 = tensor.reshape %transposed(%cst_0) {ssbuffer.block_id = 15 : i32} : (tensor<4x64x8xf32>, tensor<4xi64>) -> tensor<4x4x16x8xf32>
        hivm.hir.sync_block_wait {ssbuffer.block_id = 15 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 1
        hivm.hir.copy ins(%reshape_43 : tensor<4x4x16x8xf32>) outs(%alloc_10 : memref<4x4x16x8xf32, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 15 : i32, ssbuffer.transfer_id = 0 : i32}
        hivm.hir.sync_block_set {ssbuffer.block_id = 15 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
        hivm.hir.sync_block_wait {ssbuffer.block_id = 16 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 2
        %memspacecast_44 = memref.memory_space_cast %alloc_11 {ssbuffer.block_id = 16 : i32, ssbuffer.transfer_id = 1 : i32} : memref<64x64xf32, #hivm.address_space<ub>> to memref<64x64xf32>
        %140 = bufferization.to_tensor %memspacecast_44 restrict writable {ssbuffer.block_id = 16 : i32, ssbuffer.transfer_id = 1 : i32} : memref<64x64xf32>
        %141 = arith.addf %140, %arg43 {ssbuffer.block_id = 16 : i32} : tensor<64x64xf32>
        hivm.hir.sync_block_set {ssbuffer.block_id = 16 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 2
        scf.yield %141, %124, %125, %126 : tensor<64x64xf32>, index, index, index
      } {DataUse, ssbuffer.block_id = 24 : i32, ssbuffer.main_loop = 0 : i32}
      hivm.hir.sync_block_wait {ssbuffer.block_id = 24 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 1
      %alloc_12 = memref.alloc() {ssbuffer.block_id = 22 : i32} : memref<64xf32>
      %alloc_13 = memref.alloc() {ssbuffer.block_id = 22 : i32} : memref<64x64xf32>
      %alloc_14 = memref.alloc() {ssbuffer.block_id = 22 : i32} : memref<64x64xf32>
      scf.if %61 {
        linalg.fill {ssbuffer.block_id = 22 : i32} ins(%cst_2 : f32) outs(%alloc_12 : memref<64xf32>)
      } {hivm.unlikely_condition, ssbuffer.block_id = 22 : i32}
      scf.if %62 {
        linalg.fill {ssbuffer.block_id = 22 : i32} ins(%cst_2 : f32) outs(%alloc_13 : memref<64x64xf32>)
        linalg.fill {ssbuffer.block_id = 22 : i32} ins(%cst_2 : f32) outs(%alloc_14 : memref<64x64xf32>)
      } {hivm.unlikely_condition, ssbuffer.block_id = 22 : i32}
      %72 = linalg.fill {ssbuffer.block_id = 22 : i32} ins(%cst : f32) outs(%64 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %73 = arith.muli %12, %arg18 {ssbuffer.block_id = 22 : i32} : i64
      %74 = arith.muli %15, %arg19 {ssbuffer.block_id = 22 : i32} : i64
      %75 = arith.addi %73, %74 {ssbuffer.block_id = 22 : i32} : i64
      %76 = arith.index_cast %75 {ssbuffer.block_id = 22 : i32} : i64 to index
      %77 = arith.index_cast %arg18 {ssbuffer.block_id = 22 : i32} : i64 to index
      %78 = arith.muli %arg41, %c128_i32 {ssbuffer.block_id = 22 : i32} : i32
      %79 = arith.index_cast %78 {ssbuffer.block_id = 22 : i32} : i32 to index
      %80 = arith.addi %79, %31 {ssbuffer.block_id = 22 : i32} : index
      %reinterpret_cast_15 = memref.reinterpret_cast %arg11 to offset: [%80], sizes: [64], strides: [1] {ssbuffer.block_id = 22 : i32} : memref<?xf32> to memref<64xf32, strided<[1], offset: ?>>
      %subview_16 = memref.subview %reinterpret_cast_15[0] [%48] [1] {ssbuffer.block_id = 22 : i32} : memref<64xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
      %subview_17 = memref.subview %alloc_12[0] [%48] [1] {ssbuffer.block_id = 22 : i32} : memref<64xf32> to memref<?xf32, strided<[1]>>
      memref.copy %subview_16, %subview_17 {ssbuffer.block_id = 22 : i32} : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
      %81 = bufferization.to_tensor %alloc_12 restrict writable {ssbuffer.block_id = 22 : i32} : memref<64xf32>
      %82 = arith.muli %32, %77 {ssbuffer.block_id = 22 : i32} : index
      %83 = arith.addi %76, %82 {ssbuffer.block_id = 22 : i32} : index
      %84 = arith.addi %83, %31 {ssbuffer.block_id = 22 : i32} : index
      %reinterpret_cast_18 = memref.reinterpret_cast %arg3 to offset: [%84], sizes: [64, 64], strides: [%77, 1] {ssbuffer.block_id = 22 : i32} : memref<?xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
      %subview_19 = memref.subview %reinterpret_cast_18[0, 0] [%43, %49] [1, 1] {ssbuffer.block_id = 22 : i32} : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %subview_20 = memref.subview %alloc_13[0, 0] [%43, %49] [1, 1] {ssbuffer.block_id = 22 : i32} : memref<64x64xf32> to memref<?x?xf32, strided<[64, 1]>>
      memref.copy %subview_19, %subview_20 {ssbuffer.block_id = 22 : i32} : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[64, 1]>>
      %85 = bufferization.to_tensor %alloc_13 restrict writable {ssbuffer.block_id = 22 : i32} : memref<64x64xf32>
      %broadcasted_21 = linalg.broadcast ins(%81 : tensor<64xf32>) outs(%64 : tensor<64x64xf32>) dimensions = [0]  {ssbuffer.block_id = 22 : i32}
      %86 = arith.mulf %85, %broadcasted_21 {DataUse, ssbuffer.block_id = 22 : i32} : tensor<64x64xf32>
      %87 = arith.addf %71#0, %86 {DataUse, ssbuffer.block_id = 22 : i32} : tensor<64x64xf32>
      %88 = arith.muli %12, %arg20 {ssbuffer.block_id = 22 : i32} : i64
      %89 = arith.muli %15, %arg21 {ssbuffer.block_id = 22 : i32} : i64
      %90 = arith.addi %88, %89 {ssbuffer.block_id = 22 : i32} : i64
      %91 = arith.index_cast %90 {ssbuffer.block_id = 22 : i32} : i64 to index
      %92 = arith.index_cast %arg20 {ssbuffer.block_id = 22 : i32} : i64 to index
      %93 = arith.muli %32, %92 {ssbuffer.block_id = 22 : i32} : index
      %94 = arith.addi %91, %93 {ssbuffer.block_id = 22 : i32} : index
      %95 = arith.addi %94, %31 {ssbuffer.block_id = 22 : i32} : index
      %reinterpret_cast_22 = memref.reinterpret_cast %arg4 to offset: [%95], sizes: [64, 64], strides: [%92, 1] {ssbuffer.block_id = 22 : i32} : memref<?xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
      %subview_23 = memref.subview %reinterpret_cast_22[0, 0] [%43, %49] [1, 1] {ssbuffer.block_id = 22 : i32} : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %subview_24 = memref.subview %alloc_14[0, 0] [%43, %49] [1, 1] {ssbuffer.block_id = 22 : i32} : memref<64x64xf32> to memref<?x?xf32, strided<[64, 1]>>
      memref.copy %subview_23, %subview_24 {ssbuffer.block_id = 22 : i32} : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[64, 1]>>
      %96 = bufferization.to_tensor %alloc_14 restrict writable {ssbuffer.block_id = 22 : i32} : memref<64x64xf32>
      %97 = arith.subf %65, %96 {DataUse, ssbuffer.block_id = 22 : i32} : tensor<64x64xf32>
      %98 = math.exp %97 {DataUse, ssbuffer.block_id = 22 : i32} : tensor<64x64xf32>
      %99 = arith.addf %98, %72 {DataUse, ssbuffer.block_id = 22 : i32} : tensor<64x64xf32>
      %100 = arith.divf %72, %99 {DataUse, ssbuffer.block_id = 22 : i32} : tensor<64x64xf32>
      %101 = arith.mulf %96, %100 {DataUse, ssbuffer.block_id = 22 : i32} : tensor<64x64xf32>
      %102 = arith.mulf %87, %101 {DataUse, ssbuffer.block_id = 22 : i32} : tensor<64x64xf32>
      %103 = arith.muli %12, %arg22 {ssbuffer.block_id = 22 : i32} : i64
      %104 = arith.muli %15, %arg23 {ssbuffer.block_id = 22 : i32} : i64
      %105 = arith.addi %103, %104 {ssbuffer.block_id = 22 : i32} : i64
      %106 = arith.index_cast %105 {ssbuffer.block_id = 22 : i32} : i64 to index
      %107 = arith.index_cast %arg22 {ssbuffer.block_id = 22 : i32} : i64 to index
      %108 = arith.muli %32, %107 {ssbuffer.block_id = 22 : i32} : index
      %109 = arith.addi %106, %108 {ssbuffer.block_id = 22 : i32} : index
      %110 = arith.addi %109, %31 {ssbuffer.block_id = 22 : i32} : index
      %reinterpret_cast_25 = memref.reinterpret_cast %arg5 to offset: [%110], sizes: [64, 64], strides: [%107, 1] {ssbuffer.block_id = 22 : i32} : memref<?xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
      %extracted_slice = tensor.extract_slice %102[0, 0] [%43, %49] [1, 1] {ssbuffer.block_id = 22 : i32} : tensor<64x64xf32> to tensor<?x?xf32>
      %subview_26 = memref.subview %reinterpret_cast_25[0, 0] [%43, %49] [1, 1] {ssbuffer.block_id = 22 : i32} : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice in writable %subview_26 {ssbuffer.block_id = 22 : i32} : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %0 = arith.divsi %arg39, %c2_i32 {MixUse, ssbuffer.block_id = 6 : i32} : i32
      %1 = arith.remsi %arg39, %c2_i32 {ssbuffer.block_id = 6 : i32} : i32
      %2 = arith.index_cast %arg40 {ssbuffer.block_id = 6 : i32} : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg13 to offset: [%2], sizes: [1], strides: [1] {ssbuffer.block_id = 6 : i32} : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
      %3 = memref.load %reinterpret_cast[%c0] {ssbuffer.block_id = 6 : i32} : memref<1xi64, strided<[1], offset: ?>>
      %4 = arith.addi %2, %c1 {ssbuffer.block_id = 6 : i32} : index
      %reinterpret_cast_3 = memref.reinterpret_cast %arg13 to offset: [%4], sizes: [1], strides: [1] {ssbuffer.block_id = 6 : i32} : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
      %5 = memref.load %reinterpret_cast_3[%c0] {ssbuffer.block_id = 6 : i32} : memref<1xi64, strided<[1], offset: ?>>
      %6 = arith.muli %3, %arg18 {ssbuffer.block_id = 6 : i32} : i64
      %7 = arith.extsi %arg41 {ssbuffer.block_id = 6 : i32} : i32 to i64
      %8 = arith.muli %7, %arg19 {ssbuffer.block_id = 6 : i32} : i64
      %9 = arith.addi %6, %8 {ssbuffer.block_id = 6 : i32} : i64
      %10 = arith.index_cast %9 {ssbuffer.block_id = 6 : i32} : i64 to index
      %11 = arith.muli %0, %c64_i32 {MixUse, ssbuffer.block_id = 6 : i32} : i32
      %reinterpret_cast_4 = memref.reinterpret_cast %arg8 to offset: [%2], sizes: [1], strides: [1] {ssbuffer.block_id = 6 : i32} : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
      %12 = memref.load %reinterpret_cast_4[%c0] {ssbuffer.block_id = 6 : i32} : memref<1xi64, strided<[1], offset: ?>>
      %13 = arith.subi %5, %3 {ssbuffer.block_id = 6 : i32} : i64
      %14 = arith.muli %1, %c64_i32 {ssbuffer.block_id = 6 : i32} : i32
      %15 = arith.index_cast %14 {ssbuffer.block_id = 6 : i32} : i32 to index
      %16 = arith.index_cast %11 {ssbuffer.block_id = 6 : i32} : i32 to index
      %17 = arith.addi %16, %c64 {ssbuffer.block_id = 6 : i32} : index
      %18 = arith.index_cast %13 {ssbuffer.block_id = 6 : i32} : i64 to index
      %19 = arith.maxsi %16, %18 {ssbuffer.block_id = 6 : i32} : index
      %20 = arith.minsi %17, %19 {ssbuffer.block_id = 6 : i32} : index
      %21 = arith.subi %20, %16 {ssbuffer.block_id = 6 : i32} : index
      %22 = arith.minsi %21, %c64 {ssbuffer.block_id = 6 : i32} : index
      %23 = arith.cmpi slt, %22, %c64 {ssbuffer.block_id = 6 : i32} : index
      %24 = arith.addi %15, %c64 {ssbuffer.block_id = 6 : i32} : index
      %25 = arith.maxsi %15, %c128 {ssbuffer.block_id = 6 : i32} : index
      %26 = arith.minsi %24, %25 {ssbuffer.block_id = 6 : i32} : index
      %27 = arith.subi %26, %15 {ssbuffer.block_id = 6 : i32} : index
      %28 = arith.minsi %27, %c64 {ssbuffer.block_id = 6 : i32} : index
      %29 = arith.cmpi slt, %28, %c64 {ssbuffer.block_id = 6 : i32} : index
      %30 = arith.muli %arg18, %c32_i64 {ssbuffer.block_id = 6 : i32} : i64
      %31 = arith.index_cast %arg18 {ssbuffer.block_id = 6 : i32} : i64 to index
      %32 = arith.extsi %arg40 {ssbuffer.block_id = 20 : i32} : i32 to i64
      %33 = arith.divsi %arg39, %c2_i32 {MixUse, ssbuffer.block_id = 20 : i32} : i32
      %34 = arith.index_cast %arg40 {ssbuffer.block_id = 20 : i32} : i32 to index
      %reinterpret_cast_5 = memref.reinterpret_cast %arg13 to offset: [%34], sizes: [1], strides: [1] {ssbuffer.block_id = 20 : i32} : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
      %35 = memref.load %reinterpret_cast_5[%c0] {ssbuffer.block_id = 20 : i32} : memref<1xi64, strided<[1], offset: ?>>
      %36 = arith.addi %34, %c1 {ssbuffer.block_id = 20 : i32} : index
      %reinterpret_cast_6 = memref.reinterpret_cast %arg13 to offset: [%36], sizes: [1], strides: [1] {ssbuffer.block_id = 20 : i32} : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
      %37 = memref.load %reinterpret_cast_6[%c0] {ssbuffer.block_id = 20 : i32} : memref<1xi64, strided<[1], offset: ?>>
      %38 = arith.extsi %arg41 {ssbuffer.block_id = 20 : i32} : i32 to i64
      %reinterpret_cast_7 = memref.reinterpret_cast %arg8 to offset: [%34], sizes: [1], strides: [1] {ssbuffer.block_id = 20 : i32} : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
      %39 = memref.load %reinterpret_cast_7[%c0] {ssbuffer.block_id = 20 : i32} : memref<1xi64, strided<[1], offset: ?>>
      %40 = arith.cmpi sge, %32, %c1_i64 {ssbuffer.block_id = 20 : i32} : i64
      %41 = arith.addi %34, %c-1 {ssbuffer.block_id = 20 : i32} : index
      %reinterpret_cast_8 = memref.reinterpret_cast %arg8 to offset: [%41], sizes: [1], strides: [1] {ssbuffer.block_id = 20 : i32} : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
      %42 = memref.load %reinterpret_cast_8[%c0] {ssbuffer.block_id = 20 : i32} : memref<1xi64, strided<[1], offset: ?>>
      %43 = tensor.empty() {ssbuffer.block_id = 20 : i32} : tensor<1xi1>
      %inserted = tensor.insert %40 into %43[%c0] {ssbuffer.block_id = 20 : i32} : tensor<1xi1>
      %44 = tensor.empty() {ssbuffer.block_id = 20 : i32} : tensor<1xi64>
      %inserted_9 = tensor.insert %42 into %44[%c0] {ssbuffer.block_id = 20 : i32} : tensor<1xi64>
      %45 = linalg.fill {ssbuffer.block_id = 20 : i32} ins(%c-1_i64 : i64) outs(%44 : tensor<1xi64>) -> tensor<1xi64>
      %46 = arith.select %inserted, %inserted_9, %45 {ssbuffer.block_id = 20 : i32} : tensor<1xi1>, tensor<1xi64>
      %47 = arith.subi %37, %35 {ssbuffer.block_id = 20 : i32} : i64
      %48 = arith.addi %33, %c1_i32 {Undefined, ssbuffer.block_id = 20 : i32} : i32
      %49 = arith.muli %48, %c64_i32 {Undefined, ssbuffer.block_id = 20 : i32} : i32
      %50 = arith.extsi %49 {Undefined, ssbuffer.block_id = 20 : i32} : i32 to i64
      %51 = arith.minsi %50, %47 {Undefined, ssbuffer.block_id = 20 : i32} : i64
      %extracted = tensor.extract %46[%c0] {ssbuffer.block_id = 7 : i32} : tensor<1xi64>
      %52 = arith.cmpi ne, %12, %extracted {ssbuffer.block_id = 7 : i32} : i64
      %53 = scf.if %52 -> (i64) {
        %75 = arith.muli %39, %arg33 {ssbuffer.block_id = 8 : i32} : i64
        %76 = arith.muli %38, %arg34 {ssbuffer.block_id = 8 : i32} : i64
        %77 = arith.addi %75, %76 {ssbuffer.block_id = 8 : i32} : i64
        scf.yield {Undefined} %77 : i64
      } else {
        %75 = arith.subi %32, %c1_i64 {ssbuffer.block_id = 9 : i32} : i64
        %76 = arith.muli %75, %arg30 {ssbuffer.block_id = 9 : i32} : i64
        %77 = arith.muli %38, %arg31 {ssbuffer.block_id = 9 : i32} : i64
        %78 = arith.addi %76, %77 {ssbuffer.block_id = 9 : i32} : i64
        scf.yield {Undefined} %78 : i64
      } {ssbuffer.block_id = 23 : i32}
      %alloc = memref.alloc() {ssbuffer.block_id = 5 : i32} : memref<64x64xf32>
      %alloc_10 = memref.alloc() {ssbuffer.block_id = 5 : i32} : memref<64x64xf32>
      scf.if %29 {
        linalg.fill {ssbuffer.block_id = 5 : i32} ins(%cst_2 : f32) outs(%alloc_10 : memref<64x64xf32>)
      } {hivm.unlikely_condition, ssbuffer.block_id = 5 : i32}
      scf.if %23 {
        linalg.fill {ssbuffer.block_id = 5 : i32} ins(%cst_2 : f32) outs(%alloc : memref<64x64xf32>)
      } {hivm.unlikely_condition, ssbuffer.block_id = 5 : i32}
      %54 = tensor.empty() {ssbuffer.block_id = 5 : i32} : tensor<64x64xf32>
      %55 = arith.divsi %arg41, %c4_i32 {ssbuffer.block_id = 5 : i32} : i32
      %56 = arith.extsi %55 {ssbuffer.block_id = 5 : i32} : i32 to i64
      %57 = arith.muli %3, %arg28 {ssbuffer.block_id = 5 : i32} : i64
      %58 = arith.muli %56, %arg29 {ssbuffer.block_id = 5 : i32} : i64
      %59 = arith.addi %57, %58 {ssbuffer.block_id = 5 : i32} : i64
      %60 = arith.index_cast %59 {ssbuffer.block_id = 5 : i32} : i64 to index
      %61 = arith.select %52, %arg35, %arg32 {ssbuffer.block_id = 5 : i32} : i64
      %62 = arith.index_cast %61 {ssbuffer.block_id = 5 : i32} : i64 to index
      %63 = arith.index_cast %53 {ssbuffer.block_id = 5 : i32} : i64 to index
      %64 = arith.index_cast %arg28 {ssbuffer.block_id = 5 : i32} : i64 to index
      %65 = arith.muli %16, %64 {ssbuffer.block_id = 5 : i32} : index
      %66 = arith.addi %60, %65 {ssbuffer.block_id = 5 : i32} : index
      %reinterpret_cast_11 = memref.reinterpret_cast %arg9 to offset: [%66], sizes: [64, 64], strides: [%64, 1] {ssbuffer.block_id = 5 : i32} : memref<?xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
      %subview = memref.subview %reinterpret_cast_11[0, 0] [%22, 64] [1, 1] {ssbuffer.block_id = 5 : i32} : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x64xf32, strided<[?, 1], offset: ?>>
      %subview_12 = memref.subview %alloc[0, 0] [%22, 64] [1, 1] {ssbuffer.block_id = 5 : i32} : memref<64x64xf32> to memref<?x64xf32, strided<[64, 1]>>
      memref.copy %subview, %subview_12 {ssbuffer.block_id = 5 : i32} : memref<?x64xf32, strided<[?, 1], offset: ?>> to memref<?x64xf32, strided<[64, 1]>>
      %67 = bufferization.to_tensor %alloc restrict writable {ssbuffer.block_id = 5 : i32} : memref<64x64xf32>
      %68 = arith.muli %15, %62 {ssbuffer.block_id = 5 : i32} : index
      %69 = arith.addi %63, %68 {ssbuffer.block_id = 5 : i32} : index
      %reinterpret_cast_13 = memref.reinterpret_cast %arg12 to offset: [%69], sizes: [64, 64], strides: [%62, 1] {ssbuffer.block_id = 5 : i32} : memref<?xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
      %subview_14 = memref.subview %reinterpret_cast_13[0, 0] [%28, 64] [1, 1] {ssbuffer.block_id = 5 : i32} : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x64xf32, strided<[?, 1], offset: ?>>
      %subview_15 = memref.subview %alloc_10[0, 0] [%28, 64] [1, 1] {ssbuffer.block_id = 5 : i32} : memref<64x64xf32> to memref<?x64xf32, strided<[64, 1]>>
      memref.copy %subview_14, %subview_15 {ssbuffer.block_id = 5 : i32} : memref<?x64xf32, strided<[?, 1], offset: ?>> to memref<?x64xf32, strided<[64, 1]>>
      %70 = bufferization.to_tensor %alloc_10 restrict writable {ssbuffer.block_id = 5 : i32} : memref<64x64xf32>
      %transposed = linalg.transpose ins(%70 : tensor<64x64xf32>) outs(%54 : tensor<64x64xf32>) permutation = [1, 0]  {ssbuffer.block_id = 5 : i32}
      %71 = tensor.empty() {ssbuffer.block_id = 5 : i32} : tensor<64x64xf32>
      %72 = linalg.fill {ssbuffer.block_id = 5 : i32} ins(%cst_2 : f32) outs(%71 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %73 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 5 : i32} ins(%67, %transposed : tensor<64x64xf32>, tensor<64x64xf32>) outs(%72 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %alloc_16 = memref.alloc() {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 2 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_16 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 2 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 2 : i32} ins(%73 : tensor<64x64xf32>) outs(%alloc_16 : memref<64x64xf32, #hivm.address_space<ub>>)
      hivm.hir.sync_block_set {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 2 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 3
      %alloc_17 = memref.alloc() {ssbuffer.block_id = 24 : i32, ssbuffer.transfer_id = 0 : i32} : memref<4x4x16x8xf32, #hivm.address_space<cbuf>>
      annotation.mark %alloc_17 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 24 : i32, ssbuffer.transfer_id = 0 : i32} : memref<4x4x16x8xf32, #hivm.address_space<cbuf>>
      hivm.hir.sync_block_set {ssbuffer.block_id = 24 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 1
      %alloc_18 = memref.alloc() {ssbuffer.block_id = 24 : i32, ssbuffer.transfer_id = 1 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_18 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 24 : i32, ssbuffer.transfer_id = 1 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      %74 = scf.for %arg42 = %c0_i64 to %51 step %c32_i64 iter_args(%arg43 = %c0) -> (index)  : i64 {
        hivm.hir.sync_block_wait {ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
        %75 = hivm.hir.convert_layout %alloc_17 output_shape [64, 32] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 0 : i32} : (memref<4x4x16x8xf32, #hivm.address_space<cbuf>>) -> memref<64x32xf32, #hivm.address_space<cbuf>>
        %memspacecast = memref.memory_space_cast %75 {ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 0 : i32} : memref<64x32xf32, #hivm.address_space<cbuf>> to memref<64x32xf32>
        %76 = bufferization.to_tensor %memspacecast restrict writable {ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 0 : i32} : memref<64x32xf32>
        %77 = arith.subi %13, %arg42 {ssbuffer.block_id = 3 : i32} : i64
        %alloc_19 = memref.alloc() {ssbuffer.block_id = 3 : i32} : memref<32x64xf32>
        %78 = arith.index_cast %77 {ssbuffer.block_id = 3 : i32} : i64 to index
        %79 = arith.maxsi %78, %c0 {ssbuffer.block_id = 3 : i32} : index
        %80 = arith.minsi %79, %c32 {ssbuffer.block_id = 3 : i32} : index
        %81 = arith.minsi %80, %c32 {ssbuffer.block_id = 3 : i32} : index
        %82 = arith.cmpi slt, %81, %c32 {ssbuffer.block_id = 3 : i32} : index
        %83 = arith.ori %82, %29 {ssbuffer.block_id = 3 : i32} : i1
        scf.if %83 {
          linalg.fill {ssbuffer.block_id = 3 : i32} ins(%cst_2 : f32) outs(%alloc_19 : memref<32x64xf32>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 3 : i32}
        %84 = arith.addi %10, %arg43 {ssbuffer.block_id = 3 : i32} : index
        %85 = arith.addi %84, %15 {ssbuffer.block_id = 3 : i32} : index
        %reinterpret_cast_20 = memref.reinterpret_cast %arg3 to offset: [%85], sizes: [32, 64], strides: [%31, %c1] {ssbuffer.block_id = 3 : i32} : memref<?xf32> to memref<32x64xf32, strided<[?, ?], offset: ?>>
        %subview_21 = memref.subview %reinterpret_cast_20[0, 0] [%81, %28] [1, 1] {ssbuffer.block_id = 3 : i32} : memref<32x64xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %subview_22 = memref.subview %alloc_19[0, 0] [%81, %28] [1, 1] {ssbuffer.block_id = 3 : i32} : memref<32x64xf32> to memref<?x?xf32, strided<[64, 1]>>
        memref.copy %subview_21, %subview_22 {ssbuffer.block_id = 3 : i32} : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[64, 1]>>
        %86 = bufferization.to_tensor %alloc_19 restrict writable {ssbuffer.block_id = 3 : i32} : memref<32x64xf32>
        %87 = tensor.empty() {ssbuffer.block_id = 3 : i32} : tensor<64x64xf32>
        %88 = linalg.fill {ssbuffer.block_id = 3 : i32} ins(%cst_2 : f32) outs(%87 : tensor<64x64xf32>) -> tensor<64x64xf32>
        %89 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 3 : i32} ins(%76, %86 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%88 : tensor<64x64xf32>) -> tensor<64x64xf32>
        %90 = arith.index_cast %30 {ssbuffer.block_id = 3 : i32} : i64 to index
        %91 = arith.addi %arg43, %90 {ssbuffer.block_id = 3 : i32} : index
        hivm.hir.sync_block_set {ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 1
        hivm.hir.sync_block_wait {ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 2
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 1 : i32} ins(%89 : tensor<64x64xf32>) outs(%alloc_18 : memref<64x64xf32, #hivm.address_space<ub>>)
        hivm.hir.sync_block_set {ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 2
        scf.yield %91 : index
      } {DataUse, ssbuffer.block_id = 24 : i32, ssbuffer.main_loop = 0 : i32}
      hivm.hir.sync_block_wait {ssbuffer.block_id = 24 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 2
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return
  }
}