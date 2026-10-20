// RUN: neura-interpreter %s --dataflow --verbose | head -n 300 | FileCheck %s

// The void return is ready at the first topological level, but the adds
// become ready only one level after another. Execution must not end until
// the last add has run. head bounds the output if execution never ends.

func.func @void_return_with_late_ops() {
  %c = "neura.constant"() <{value = 1 : i32}> : () -> i32
  %a = "neura.add"(%c, %c) : (i32, i32) -> i32
  %b = "neura.add"(%a, %a) : (i32, i32) -> i32
  %d = "neura.add"(%b, %b) : (i32, i32) -> i32
  return
}

// CHECK:     → Output: (void)
// CHECK:     Topological Level 4
// CHECK:     Result  : value = 8.000000e+00
// CHECK:     Execution terminated after void return at end of DFG iteration 0
// CHECK-NOT: DFG Iteration 1
