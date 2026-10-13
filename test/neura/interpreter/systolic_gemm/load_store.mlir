// RUN: neura-interpreter %s --dataflow --verbose | FileCheck %s

// Smoke test for the configured (memref + constants) form of neura.load /
// neura.store, without neura.mac: load an element from %a, add 1, store it
// back. The three-element constants array also exercises the trip-count
// derivation from Step 02 (constants.size() == 3 -> the whole load/add/store
// chain fires 3 times, once per offset).

func.func @test_load_add_store(%a: memref<3xi32>) {
  neura.kernel inputs(%a : memref<3xi32>) attributes {
    interp.init_values = [dense<[10, 20, 30]> : tensor<3xi32>]
  } {
  ^bb0(%arg: memref<3xi32>):
    %one = arith.constant 1 : i32
    %v = "neura.load"(%arg) {constants = array<i64: 0, 1, 2>}
         : (memref<3xi32>) -> i32
    %v1 = "neura.add"(%v, %one) : (i32, i32) -> i32
    "neura.store"(%v1, %arg) {constants = array<i64: 0, 1, 2>}
         : (i32, memref<3xi32>) -> ()
    neura.yield
  }
  return
}

// CHECK: Derived trip count = 3 from load/store constants
// CHECK: Loaded element 0 (firing #0) = 1.000000e+01
// CHECK: Stored element 0 (firing #0) = 11
// CHECK: Loaded element 1 (firing #1) = 2.000000e+01
// CHECK: Stored element 1 (firing #1) = 21
// CHECK: Loaded element 2 (firing #2) = 3.000000e+01
// CHECK: Stored element 2 (firing #2) = 31
// CHECK: Final memory[input #0] = [11, 21, 31]
