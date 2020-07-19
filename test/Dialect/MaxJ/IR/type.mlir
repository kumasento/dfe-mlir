// RUN: dfe-opt %s -mlir-print-op-generic | dfe-opt | dfe-opt | FileCheck %s

// CHECK: maxj.kernel @check_typecast () -> () {
maxj.kernel @check_typecast () -> () {
  // CHECK-NEXT: %[[C0:.*]] = maxj.const {{.*}} : f64 -> !maxj.svar<f64>
  %0 = maxj.const 1.0 : f64 -> !maxj.svar<f64>
  // CHECK-NEXT: %[[C1:.*]] = maxj.cast %[[C0]] : !maxj.svar<f64> -> !maxj.svar<i32>
  %1 = maxj.cast %0 : !maxj.svar<f64> -> !maxj.svar<i32>
// CHECK-NEXT: }
}