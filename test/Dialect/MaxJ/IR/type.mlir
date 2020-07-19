// RUN: dfe-opt %s -mlir-print-op-generic | dfe-opt | dfe-opt | FileCheck %s

// CHECK: maxj.kernel @typecast () -> () {
maxj.kernel @typecast () -> () {
  %0 = maxj.const 1.0 : f64 -> !maxj.svar<f64>
  %1 = maxj.cast %0 : !maxj.svar<f64> -> !maxj.svar<i32>
}