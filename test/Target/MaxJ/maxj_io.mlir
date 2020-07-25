// RUN: dfe-translate --maxj-to-maxj %s | FileCheck %s

// CHECK: public class ScalarIO extends Kernel {
// CHECK-NEXT: public ScalarIO(KernelParameters params) {
// CHECK-NEXT: super(params);
maxj.kernel @ScalarIO () -> () {
  // CHECK-NEXT: DFEVar _0 = io.input("a0", dfeInt(32));
  %0 = maxj.input "a0" -> !maxj.svar<i32>
  // CHECK-NEXT: io.output("b0", _0, dfeInt(32));
  maxj.output "b0", %0 : !maxj.svar<i32>

  // CHECK-NEXT: bool _1 = true;
  %c0 = constant true
  // CHECK-NEXT: DFEVar _2 = constant.var(dfeInt(1), _1);
  %e1 = maxj.const %c0 : !maxj.svar<i1>
  // CHECK-NEXT: DFEVar _3 = io.input("a1", dfeInt(8), _2);
  %1 = maxj.input "a1", %e1 : !maxj.svar<i1> -> !maxj.svar<i8>
  // CHECK-NEXT: io.output("b1", _3, dfeInt(8), _2);
  maxj.output "b1", %1 : !maxj.svar<i8>, %e1 : !maxj.svar<i1>
  // CHECK-NEXT: }
  // CHECK-NEXT: }
}

// CHECK: public class VectorIO extends Kernel {
// CHECK-NEXT: public VectorIO(KernelParameters params) {
// CHECK-NEXT: super(params);
maxj.kernel @VectorIO () -> () {
  // CHECK-NEXT: DFEVector<DFEVar> _0 = io.input("in_v0", (new DFEVectorType<DFEVar>(dfeInt(32), 8)));
  %v0 = maxj.input "in_v0" -> !maxj.svar<vector<8xi32>>
  // CHECK-NEXT: io.output("out_v0", _0, (new DFEVectorType<DFEVar>(dfeInt(32), 8)));
  maxj.output "out_v0", %v0 : !maxj.svar<vector<8xi32>>
// CHECK-NEXT: }
// CHECK-NEXT: }
}