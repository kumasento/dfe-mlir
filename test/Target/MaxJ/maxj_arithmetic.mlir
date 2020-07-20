// RUN: dfe-translate --maxj-to-maxj %s | FileCheck %s

// CHECK: public class ScalarArith extends Kernel {
// CHECK-NEXT: public ScalarArith(KernelParameters params) {
// CHECK-NEXT: super(params);
maxj.kernel @ScalarArith() -> () {
  // CHECK-NEXT: DFEVar _0 = io.input("A", dfeInt(32));
  %0 = maxj.input "A" -> !maxj.svar<i32>
  // CHECK-NEXT: DFEVar _1 = io.input("B", dfeInt(32));
  %1 = maxj.input "B" -> !maxj.svar<i32>
  // CHECK-NEXT: DFEVar _2 = _0 + _1;
  %2 = maxj.add %0, %1 : !maxj.svar<i32>
  // CHECK-NEXT: DFEVar _3 = _0 * _1;
  %3 = maxj.mul %0, %1 : !maxj.svar<i32>
// CHECK-NEXT: }
// CHECK-NEXT: }
}