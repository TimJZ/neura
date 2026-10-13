// RUN: neura-interpreter %s --dataflow --verbose | FileCheck %s

// Minimal smoke test for entering neura.kernel: no load/store/mac, just
// verifies block-argument binding (memref input gets a backing buffer,
// seeded from interp.init_values) and the trip-count fallback (no
// load/store in the body -> runs once).

func.func @test_kernel_entry(%a: memref<2x2xi32>) {
  neura.kernel inputs(%a : memref<2x2xi32>) attributes {
    interp.init_values = [dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>]
  } {
  ^bb0(%arg: memref<2x2xi32>):
    neura.yield
  }
  return
}

// CHECK: Entering neura.kernel
// CHECK: Bound input #0 to memref buffer (4 elements)
// CHECK: No load/store found in kernel body; trip count = 1
