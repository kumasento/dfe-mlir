// RUN: dfe-opt %s -mlir-print-op-generic | dfe-opt | dfe-opt | FileCheck %s

// CHECK: maxj.kernel @check_read () -> () {
maxj.kernel @check_read() -> () {
  // CHECK-NEXT: %[[C0:.*]] = maxj.alloc() : !maxj.mem<128xi32>
  %0 = maxj.alloc() : !maxj.mem<128xi32>
  // CHECK-NEXT: %[[C1:.*]] = maxj.counter 8 : i64 -> !maxj.svar<i8>
  %1 = maxj.counter 8: i64 -> !maxj.svar<i8>
  // CHECK-NEXT: %[[C2:.*]] = maxj.read %[[C0]] : !maxj.mem<128xi32>, %[[C1]] : !maxj.svar<i8>
  %2 = maxj.read %0 : !maxj.mem<128xi32>, %1: !maxj.svar<i8>
// CHECK-NEXT: }
}

// CHECK: maxj.kernel @check_write () -> () {
maxj.kernel @check_write() -> () {
  // CHECK-NEXT: %[[C0:.*]] = maxj.alloc() : !maxj.mem<128xi32>
  %0 = maxj.alloc() : !maxj.mem<128xi32>
  // CHECK-NEXT: %[[C1:.*]] = maxj.counter 8 : i64 -> !maxj.svar<i8>
  %1 = maxj.counter 8 : i64 -> !maxj.svar<i8>
  // CHECK-NEXT: %[[C2:.*]] = maxj.input "val" -> !maxj.svar<i32>
  %2 = maxj.input "val" -> !maxj.svar<i32>
  // CHECK-NEXT: %[[C3:.*]] = maxj.input "enable" -> !maxj.svar<i1>
  %3 = maxj.input "enable" -> !maxj.svar<i1>
  // CHECK-NEXT: maxj.write %[[C0]] : !maxj.mem<128xi32>, %[[C1]] : !maxj.svar<i8>, %[[C2]] : !maxj.svar<i32>, %[[C3]] : !maxj.svar<i1>
  maxj.write %0 : !maxj.mem<128xi32>, %1 : !maxj.svar<i8>, %2 : !maxj.svar<i32>, %3 : !maxj.svar<i1>
// CHECK-NEXT: }
}