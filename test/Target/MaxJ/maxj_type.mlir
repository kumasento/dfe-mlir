// RUN: dfe-translate --maxj-to-maxj %s | FileCheck %s

// All type related operations

// CHECK: public class Typecast extends Kernel {
// CHECK-NEXT: public Typecast(KernelParameters params) {
// CHECK-NEXT: super(params);
maxj.kernel @Typecast () -> () {
  // CHECK-NEXT: double _0 = {{.*}};
  %c0 = constant 1.0 : f64
  // CHECK-NEXT: DFEVar _1 = constant.var(dfeFloat(11, 52), _0);
  %0 = maxj.const %c0 : !maxj.svar<f64>
  // CHECK-NEXT: DFEVar _2 = _1.cast(dfeInt(32));
  %1 = maxj.cast %0 : !maxj.svar<f64> -> !maxj.svar<i32>
// CHECK-NEXT: }
// CHECK-NEXT: }
}