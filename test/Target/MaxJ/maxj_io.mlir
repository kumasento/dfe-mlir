// RUN: dfe-translate --maxj-to-maxj %s | FileCheck %s

// CHECK: public class ScalarIO extends Kernel {
// CHECK-NEXT: public ScalarIO(KernelParameters params) {
// CHECK-NEXT: super(params);
maxj.kernel @ScalarIO () -> () {
  // CHECK-NEXT: DFEVar _0 = io.input("a0", dfeInt(32));
  %0 = maxj.input "a0" -> !maxj.svar<i32>
  // CHECK-NEXT: io.output("b0", _0, dfeInt(32));
  maxj.output "b0", %0 : !maxj.svar<i32>

  // CHECK-NEXT: DFEVar _1 = constant.var(dfeInt(1), {{.*}});
  %e1 = maxj.const 1. : f64 -> !maxj.svar<i1>
  // CHECK-NEXT: DFEVar _2 = io.input("a1", dfeInt(8), _1);
  %1 = maxj.input "a1", %e1 : !maxj.svar<i1> -> !maxj.svar<i8>
  // CHECK-NEXT: io.output("b1", _2, dfeInt(8), _1);
  maxj.output "b1", %1 : !maxj.svar<i8>, %e1 : !maxj.svar<i1>
  // CHECK-NEXT: }
  // CHECK-NEXT: }
}