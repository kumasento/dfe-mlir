// RUN: dfe-opt %s -mlir-print-op-generic | dfe-opt | dfe-opt | FileCheck %s

// CHECK: maxj.kernel @check_counter () -> () {
maxj.kernel @check_counter() -> () {
  // CHECK-NEXT: %{{.*}} = maxj.counter 1 : i64 -> !maxj.svar<i1>
  %0 = maxj.counter 1 : i64 -> !maxj.svar<i1>
  // CHECK-NEXT: %{{.*}} = maxj.counter 8 : i64, 32 : i64 -> !maxj.svar<i8>
  %1 = maxj.counter 8 : i64, 32 : i64 -> !maxj.svar<i8>
// CHECK-NEXT: }
}