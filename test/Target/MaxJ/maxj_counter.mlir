// RUN: dfe-translate --maxj-to-maxj %s | FileCheck %s

// CHECK: public class SimpleCounter extends Kernel {
// CHECK-NEXT: public SimpleCounter(KernelParameters params) {
// CHECK-NEXT: super(params);
maxj.kernel @SimpleCounter() -> () {
  // CHECK-NEXT: DFEVar _0 = control.count.simpleCounter(1);
  %0 = maxj.counter 1 : i64 -> !maxj.svar<i1>
  // CHECK-NEXT: DFEVar _1 = control.count.simpleCounter(8, 32);
  %1 = maxj.counter 8 : i64, 32 : i64 -> !maxj.svar<i8>
// CHECK-NEXT: }
// CHECK-NEXT: }
}