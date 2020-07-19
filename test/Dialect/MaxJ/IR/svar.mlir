// RUN: dfe-opt %s -mlir-print-op-generic | dfe-opt | dfe-opt | FileCheck %s

// CHECK: maxj.kernel @check_svar () -> () {
maxj.kernel @check_svar() -> () {
  // CHECK-NEXT: %{{.*}} = maxj.svar : i1
  %0 = maxj.svar : i1 
  // CHECK-NEXT: %{{.*}} = maxj.svar : vector<4xi8>
  %1 = maxj.svar : vector<4xi8>
// CHECK-NEXT: }
}

// CHECK: maxj.kernel @check_offset () -> () {
maxj.kernel @check_offset() -> () {
  // CHECK-NEXT: %[[C0:.*]] = maxj.input "a" -> !maxj.svar<i32>
  %0 = maxj.input "a" -> !maxj.svar<i32>
  // CHECK-NEXT: %[[C1:.*]] = maxj.offset -1 : i64, %[[C0]] : !maxj.svar<i32>
  %1 = maxj.offset -1 : i64, %0 : !maxj.svar<i32>
  // CHECK-NEXT: %[[V0:.*]] = maxj.input "vec_a" -> !maxj.svar<vector<8xi32>>
  %v0 = maxj.input "vec_a" -> !maxj.svar<vector<8xi32>>
  // CHECK-NEXT: %[[V1:.*]] = maxj.offset -1 : i64, %[[V0]] : !maxj.svar<vector<8xi32>>
  %v1 = maxj.offset -1 : i64, %v0 : !maxj.svar<vector<8xi32>>
// CHECK-NEXT: }
}

// CHECK: maxj.kernel @check_conn () -> () {
maxj.kernel @check_conn() -> () {
  // CHECK-NEXT: %[[C0:.*]] = maxj.svar : i32
  %0 = maxj.svar : i32
  // CHECK-NEXT: %[[C1:.*]] = maxj.input "a" -> !maxj.svar<i32>
  %1 = maxj.input "a" -> !maxj.svar<i32>
  // CHECK-NEXT: maxj.conn %[[C1]], %[[C0]] : !maxj.svar<i32>
  maxj.conn %1, %0 : !maxj.svar<i32>
// CHECK-NEXT: }
}