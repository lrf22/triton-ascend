// RUN: triton-opt --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync --mark-main-loop %s | FileCheck %s

module {
    func.func @test_iterarg_dependencies_yield_conflict(%arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}){
    %c1_i32 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} 1 : i32
    %c128_i32 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} 128 : i32
    %c0_i32 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} 0 : i32
    %cst_0 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} 0.000000e+00 : f32
    %2 = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
    %3 = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst_0 : f32) outs(%2 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %4 = math.exp %3 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>

    %cst_10 = arith.constant {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} 0.000000e+00 : f32
    %20 = tensor.empty() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
    %30 = linalg.fill {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_10 : f32) outs(%20 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %100 = tensor.empty() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
    %10 = linalg.fill {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_10 : f32) outs(%100 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %40 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%30, %30 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%10 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %0 = tensor.empty() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
    %1 = linalg.fill {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_0 : f32) outs(%0 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %91:2 = scf.for %arg20 = %c0_i32 to %c128_i32 step %c1_i32 iter_args(%arg21 = %4, %arg22 = %40) -> (tensor<32x32xf32>, tensor<32x32xf32>)  : i32 {
        %6 = math.exp %arg21 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>

        %182 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} ins(%arg22, %arg22 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%1 : tensor<32x32xf32>) -> tensor<32x32xf32>

        scf.yield {ssbuffer.core_type = "CUBE, VECTOR"} %182, %6 : tensor<32x32xf32>, tensor<32x32xf32>
    } {ssbuffer.core_type = "CUBE, VECTOR"}
    return
}}

// CHECK-LABEL: func.func @test_iterarg_dependencies_yield_conflict

// CHECK: %2 = math.exp %1 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>

// CHECK: %7 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%4, %4 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%6 : tensor<32x32xf32>) -> tensor<32x32xf32>

// CHECK: %[[ALLOC:[a-z0-9_]+]] = memref.alloc() {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32} : memref<4x2x16x8xf32, #hivm.address_space<cbuf>>
// CHECK: annotation.mark %[[ALLOC]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32} : memref<4x2x16x8xf32, #hivm.address_space<cbuf>>
// CHECK: %[[ALLOC_1:[a-z0-9_]+]] = memref.alloc() {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : memref<4x2x16x8xf32, #hivm.address_space<cbuf>>
// CHECK: annotation.mark %[[ALLOC_1]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : memref<4x2x16x8xf32, #hivm.address_space<cbuf>>
// CHECK: hivm.hir.sync_block_set {{.*}} flag = 1

// CHECK: %[[ALLOC_2:[a-z0-9_]+]] = memref.alloc() {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 1 : i32} : memref<32x32xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %[[ALLOC_2]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 1 : i32} : memref<32x32xf32, #hivm.address_space<ub>>
// CHECK: %[[ALLOC_3:[a-z0-9_]+]] = memref.alloc() {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 1 : i32} : memref<32x32xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %[[ALLOC_3]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 1 : i32} : memref<32x32xf32, #hivm.address_space<ub>>
// CHECK: hivm.hir.sync_block_set {{.*}} flag = 2

// CHECK: scf.for {{.*}} iter_args(%[[ARG2:[a-z0-9_]+]] = %2, %[[ARG3:[a-z0-9_]+]] = %7)
// CHECK: %[[EXP_11:[a-z0-9_]+]] = math.exp %[[ARG2]] {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
// CHECK: arith.constant
// CHECK: tensor.reshape %[[EXP_11]]({{.*}}
// CHECK: tensor.empty()
// CHECK: linalg.transpose
// CHECK: arith.constant
// CHECK: %[[RESHAPE_6:[a-z0-9_]+]] = tensor.reshape
// CHECK: hivm.hir.sync_block_wait {{.*}} flag = 1
// CHECK: hivm.hir.copy ins(%[[RESHAPE_6]] : tensor<4x2x16x8xf32>) outs(%[[ALLOC]] : memref<4x2x16x8xf32, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32}
// CHECK: hivm.hir.sync_block_set {{.*}} flag = 1

// CHECK: %[[MATMUL_13:[a-z0-9_]+]] = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} ins(%[[ARG3]], %[[ARG3]] : tensor<32x32xf32>, tensor<32x32xf32>)
// CHECK: hivm.hir.sync_block_wait {{.*}} flag = 2
// CHECK: hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 1 : i32} ins(%[[MATMUL_13]] : tensor<32x32xf32>) outs(%[[ALLOC_2]] : memref<32x32xf32, #hivm.address_space<ub>>)
// CHECK: hivm.hir.sync_block_set {{.*}} flag = 2
// CHECK: 
// CHECK: hivm.hir.sync_block_wait {{.*}} flag = 2
// CHECK: memref.memory_space_cast %[[ALLOC_3]] {{{.*}}
// CHECK: %[[TENSOR_14:[a-z0-9_]+]] = bufferization.to_tensor
// CHECK: arith.constant {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "VECTOR"} 0 : i32
// CHECK: hivm.hir.sync_block_set {{.*}} flag = 2

// CHECK: hivm.hir.sync_block_wait {{.*}} flag = 1
// CHECK: hivm.hir.convert_layout %[[ALLOC_1]] output_shape {{.*}}
// CHECK: memref.memory_space_cast
// CHECK: %[[TENSOR_16:[a-z0-9_]+]] = bufferization.to_tensor
// CHECK: hivm.hir.sync_block_set {{.*}} flag = 1

// CHECK: scf.yield {ssbuffer.core_type = "VECTOR, CUBE"} %[[TENSOR_14]], %[[TENSOR_16]] : tensor<32x32xf32>, tensor<32x32xf32>
// CHECK: } {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR, CUBE", ssbuffer.main_loop = 0 : i32}
// CHECK: hivm.hir.sync_block_wait {{.*}} flag = 2
// CHECK: hivm.hir.sync_block_wait {{.*}} flag = 1
// CHECK: return

