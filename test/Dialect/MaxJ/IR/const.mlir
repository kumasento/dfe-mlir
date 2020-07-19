// RUN: dfe-opt %s -mlir-print-op-generic | dfe-opt | dfe-opt | FileCheck %s

maxj.kernel @check_const() -> () {
  // a boolean (looks a bit weird)
  // CHECK: %{{.*}} = maxj.const {{.*}} : f64 -> !maxj.svar<i1>
  %0 = maxj.const 1. : f64 -> !maxj.svar<i1>
  // CHECK-NEXT: %{{.*}} = maxj.const {{.*}} : f64 -> !maxj.svar<f64>
  %1 = maxj.const 0.283: f64 -> !maxj.svar<f64>
  // CHECK-NEXT: %{{.*}} = maxj.const {{.*}} : f64 -> !maxj.svar<f32>
  %2 = maxj.const 1. : f64 -> !maxj.svar<f32>
}