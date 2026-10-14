// RUN: split-file %s %t
// RUN: not neura-interpreter %t/scalar_input.mlir --dataflow 2>&1 | FileCheck %t/scalar_input.mlir
// RUN: not neura-interpreter %t/kernel_result.mlir --dataflow 2>&1 | FileCheck %t/kernel_result.mlir
// RUN: not neura-interpreter %t/memref_type.mlir --dataflow 2>&1 | FileCheck %t/memref_type.mlir
// RUN: not neura-interpreter %t/init_size.mlir --dataflow 2>&1 | FileCheck %t/init_size.mlir
// RUN: not neura-interpreter %t/init_count.mlir --dataflow 2>&1 | FileCheck %t/init_count.mlir
// RUN: not neura-interpreter %t/trip_count.mlir --dataflow 2>&1 | FileCheck %t/trip_count.mlir

// neura.kernel inputs and forms that the interpreter does not support give an
// error instead of a wrong result.

//--- scalar_input.mlir
// In dataflow mode, kernels run before the other operations of the function.
func.func @scalar_input() {
  %c = "neura.constant"() <{value = 1 : i32}> : () -> i32
  neura.kernel inputs(%c : i32) {
  ^bb0(%arg: i32):
    neura.yield
  }
  return
}
// CHECK: neura.kernel: scalar input #0 has no value

//--- kernel_result.mlir
func.func @kernel_result(%a: memref<1xi32>) {
  %r = neura.kernel inputs(%a : memref<1xi32>) {
  ^bb0(%arg: memref<1xi32>):
    %v = "neura.load"(%arg) {constants = array<i64: 0>} : (memref<1xi32>) -> i32
    neura.yield results(%v : i32)
  } : i32
  return
}
// CHECK: neura.kernel: results and iter_args_init are not supported

//--- memref_type.mlir
func.func @memref_type(%a: memref<2xf32>) {
  neura.kernel inputs(%a : memref<2xf32>) {
  ^bb0(%arg: memref<2xf32>):
    neura.yield
  }
  return
}
// CHECK: neura.kernel: memref inputs must be static identity-layout i32 memrefs

//--- init_size.mlir
func.func @init_size(%a: memref<3xi32>) {
  neura.kernel inputs(%a : memref<3xi32>) attributes {
    interp.init_values = [dense<[1, 2]> : tensor<2xi32>]
  } {
  ^bb0(%arg: memref<3xi32>):
    neura.yield
  }
  return
}
// CHECK: neura.kernel: interp.init_values entry must be a dense integer attribute with 3 elements

//--- init_count.mlir
func.func @init_count(%a: memref<1xi32>) {
  neura.kernel inputs(%a : memref<1xi32>) attributes {
    interp.init_values = [dense<[1]> : tensor<1xi32>, dense<[2]> : tensor<1xi32>]
  } {
  ^bb0(%arg: memref<1xi32>):
    neura.yield
  }
  return
}
// CHECK: neura.kernel: interp.init_values has more entries than memref inputs

//--- trip_count.mlir
func.func @trip_count(%a: memref<3xi32>) {
  neura.kernel inputs(%a : memref<3xi32>) {
  ^bb0(%arg: memref<3xi32>):
    %v = "neura.load"(%arg) {constants = array<i64: 0, 1>} : (memref<3xi32>) -> i32
    "neura.store"(%v, %arg) {constants = array<i64: 0, 1, 2>} : (i32, memref<3xi32>) -> ()
    neura.yield
  }
  return
}
// CHECK: neura.kernel: load/store constants arrays disagree on trip count
