// RUN: dfe-opt %s -mlir-print-op-generic | dfe-opt | dfe-opt | FileCheck %s

// CHECK: maxj.kernel @arith () -> () {
maxj.kernel @arith() -> () {
  %0 = maxj.svar : i32
  %1 = maxj.svar : i32
  %2 = maxj.add %0, %1 : !maxj.svar<i32>
  %3 = maxj.mul %0, %1 : !maxj.svar<i32>
}