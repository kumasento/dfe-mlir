// RUN: dfe-opt %s -mlir-print-op-generic | dfe-opt | dfe-opt | FileCheck %s

// CHECK: maxj.kernel @arith () -> () {
maxj.kernel @arith() -> () {
  // CHECK-NEXT: %[[C0:.*]] = maxj.svar : i32
  %0 = maxj.svar : i32
  // CHECK-NEXT: %[[C1:.*]] = maxj.svar : i32
  %1 = maxj.svar : i32
  // CHECK-NEXT: %[[C2:.*]] = maxj.add %[[C0]], %[[C1]] : !maxj.svar<i32>
  %2 = maxj.add %0, %1 : !maxj.svar<i32>
  // CHECK-NEXT: %[[C3:.*]] = maxj.mul %[[C0]], %[[C1]] : !maxj.svar<i32>
  %3 = maxj.mul %0, %1 : !maxj.svar<i32>
// CHECK-NEXT: }
}