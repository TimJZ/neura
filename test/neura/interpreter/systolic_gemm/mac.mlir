// RUN: neura-interpreter %s --dataflow --verbose | FileCheck %s

// Smoke test for neura.mac + stationary resolution, isolated from the full
// 3x3/4x4 GEMM: a single PE at placement (0, 0), stationary weight 3 read
// from %w via the identity affine map, input 5 from %a. No input1, so this
// is the first-row (pure multiply) case: 5 * 3 + 0 = 15.

func.func @test_single_mac(%a: memref<1xi32>, %w: memref<1xi32>, %c: memref<1xi32>) {
  neura.kernel inputs(%a, %w, %c : memref<1xi32>, memref<1xi32>, memref<1xi32>)
    attributes {
      interp.init_values = [dense<[5]> : tensor<1xi32>, dense<[3]> : tensor<1xi32>],
      kernel_metadata = {kind = "template", template = {name = "systolic_array",
        stationary = {kernel_input = 1 : i32, map = affine_map<(x, y) -> (0)>}}}
    } {
  ^bb0(%arg_a: memref<1xi32>, %arg_w: memref<1xi32>, %arg_c: memref<1xi32>):
    %v = "neura.load"(%arg_a) {constants = array<i64: 0>} : (memref<1xi32>) -> i32
    %result, %forwarded = "neura.mac"(%v) {placement = {x = 0 : i32, y = 0 : i32}}
                           : (i32) -> (i32, i32)
    "neura.store"(%result, %arg_c) {constants = array<i64: 0>} : (i32, memref<1xi32>) -> ()
    neura.yield
  }
  return
}

// CHECK: input0 = 5.000000e+00, stationary = 3, input1 = 0
// CHECK: result = 1.500000e+01, forwarded = 5.000000e+00
// CHECK: Stored element 0 (firing #0) = 15
// CHECK: Final memory[input #2] = [15]
