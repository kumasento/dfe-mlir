// RUN: dfe-opt %s -mlir-print-op-generic | dfe-opt | dfe-opt | FileCheck %s

// constantly take in a value and pass it to the output.
// CHECK: maxj.kernel @foo () -> () {
maxj.kernel @foo () -> () {
  // CHECK-NEXT: %[[C0:.*]] = maxj.const
  %0 = maxj.const 1.0 : f64 -> !maxj.svar<i1>
  // CHECK-NEXT: %[[C1:.*]] = maxj.input "a", %[[C0]] : !maxj.svar<i1> -> !maxj.svar<f64>
  %1 = maxj.input "a", %0 : !maxj.svar<i1> -> !maxj.svar<f64>
  // CHECK-NEXT: maxj.output "b", %[[C1]] : !maxj.svar<f64>, %[[C0]] : !maxj.svar<i1>
  maxj.output "b", %1 : !maxj.svar<f64>, %0 : !maxj.svar<i1>
// CHECK-NEXT: }
}
