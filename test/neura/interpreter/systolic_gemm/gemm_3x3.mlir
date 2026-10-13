// RUN: neura-interpreter %s --dataflow --verbose | FileCheck %s

// Full integration test for issue #351: runs the unmapped kernel body of
// test/neura/template-mapping/weight_stationary_gemm_4x4.mlir (a 3x3
// weight-stationary GEMM on a 4x4 CGRA -- one load column, one store row,
// the remaining 3x3 Tiles doing neura.mac) through the interpreter and
// checks all nine outputs.
//
// Inputs and expected output match the VectorCGRA RTL test
// (tancheng/VectorCGRA@neura-systolic-integration,
// cgra/test/CgraRTL_neura_systolic_test.py), so the RTL and interpreter
// paths independently validate the same generated kernel against the same
// reference numbers.

module {
  func.func @ws_gemm_3x3(%A: memref<3x3xi32>, %B: memref<3x3xi32>, %C: memref<3x3xi32>) {
    neura.kernel inputs(%A, %B, %C : memref<3x3xi32>, memref<3x3xi32>, memref<3x3xi32>) attributes {
      accelerator = "neura",
      kernel_metadata = {kind = "template", template = {
        name = "systolic_array",
        stationary = {kernel_input = 1 : i32, map = affine_map<(x, y) -> (3 - y, x - 1)>}
      }},
      interp.init_values = [
        dense<[[1, -2, 3], [4, 0, -1], [2, 5, 1]]> : tensor<3x3xi32>,
        dense<[[2, 1, -1], [0, 3, 4], [-2, 5, 2]]> : tensor<3x3xi32>
      ]
    } {
    ^bb0(%a: memref<3x3xi32>, %b: memref<3x3xi32>, %c: memref<3x3xi32>):
      %a0 = "neura.load"(%a) {constants = array<i64: 0, 3, 6>, placement = {x = 0 : i32, y = 3 : i32}} : (memref<3x3xi32>) -> i32
      %m_1_3:2 = "neura.mac"(%a0) {placement = {x = 1 : i32, y = 3 : i32}} : (i32) -> (i32, i32)
      %m_2_3:2 = "neura.mac"(%m_1_3#1) {placement = {x = 2 : i32, y = 3 : i32}} : (i32) -> (i32, i32)
      %m_3_3:2 = "neura.mac"(%m_2_3#1) {placement = {x = 3 : i32, y = 3 : i32}} : (i32) -> (i32, i32)

      %a1 = "neura.load"(%a) {constants = array<i64: 1, 4, 7>, placement = {x = 0 : i32, y = 2 : i32}} : (memref<3x3xi32>) -> i32
      %m_1_2:2 = "neura.mac"(%a1, %m_1_3#0) {placement = {x = 1 : i32, y = 2 : i32}} : (i32, i32) -> (i32, i32)
      %m_2_2:2 = "neura.mac"(%m_1_2#1, %m_2_3#0) {placement = {x = 2 : i32, y = 2 : i32}} : (i32, i32) -> (i32, i32)
      %m_3_2:2 = "neura.mac"(%m_2_2#1, %m_3_3#0) {placement = {x = 3 : i32, y = 2 : i32}} : (i32, i32) -> (i32, i32)

      %a2 = "neura.load"(%a) {constants = array<i64: 2, 5, 8>, placement = {x = 0 : i32, y = 1 : i32}} : (memref<3x3xi32>) -> i32
      %m_1_1:2 = "neura.mac"(%a2, %m_1_2#0) {placement = {x = 1 : i32, y = 1 : i32}} : (i32, i32) -> (i32, i32)
      %m_2_1:2 = "neura.mac"(%m_1_1#1, %m_2_2#0) {placement = {x = 2 : i32, y = 1 : i32}} : (i32, i32) -> (i32, i32)
      %m_3_1:2 = "neura.mac"(%m_2_1#1, %m_3_2#0) {placement = {x = 3 : i32, y = 1 : i32}} : (i32, i32) -> (i32, i32)

      "neura.store"(%m_1_1#0, %c) {constants = array<i64: 0, 3, 6>, placement = {x = 1 : i32, y = 0 : i32}} : (i32, memref<3x3xi32>) -> ()
      "neura.store"(%m_2_1#0, %c) {constants = array<i64: 1, 4, 7>, placement = {x = 2 : i32, y = 0 : i32}} : (i32, memref<3x3xi32>) -> ()
      "neura.store"(%m_3_1#0, %c) {constants = array<i64: 2, 5, 8>, placement = {x = 3 : i32, y = 0 : i32}} : (i32, memref<3x3xi32>) -> ()
      neura.yield
    }
    return
  }
}

// Row-major C = A @ B: [-4, 10, -3, 10, -1, -6, 2, 22, 20].
// CHECK: Derived trip count = 3 from load/store constants
// CHECK: Final memory[input #2] = [-4, 10, -3, 10, -1, -6, 2, 22, 20]
