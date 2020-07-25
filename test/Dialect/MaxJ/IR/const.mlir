// RUN: dfe-opt %s -mlir-print-op-generic | dfe-opt | dfe-opt | FileCheck %s

// CHECK: maxj.kernel @check_const () -> () {
maxj.kernel @check_const() -> () {
  // CHECK-NEXT: %[[C0:.*]] = constant true
  %c0 = constant true
  // CHECK-NEXT: %[[V0:.*]] = maxj.const %[[C0]] : !maxj.svar<i1>
  %0 = maxj.const %c0 : !maxj.svar<i1>
  // CHECK-NEXT: %[[C1:.*]] = constant {{.*}} : f64
  %c1 = constant 0.283 : f64
  // CHECK-NEXT: %[[V1:.*]] = maxj.const %[[C1]] : !maxj.svar<f64>
  %1 = maxj.const %c1: !maxj.svar<f64>
  // CHECK-NEXT: %[[C2:.*]] = constant {{.*}} : f32
  %c2 = constant -3.72 : f32
  // CHECK-NEXT: %[[V2:.*]] = maxj.const %[[C2]] : !maxj.svar<f32>
  %2 = maxj.const %c2 : !maxj.svar<f32>
}
