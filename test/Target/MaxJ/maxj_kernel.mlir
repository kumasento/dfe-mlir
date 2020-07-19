// RUN: dfe-translate --maxj-to-maxj %s | FileCheck %s

// CHECK: public class empty extends Kernel {
maxj.kernel @empty () -> () {
// CHECK-NEXT: public empty(KernelParameters params) {
// CHECK-NEXT: super(params);
// CHECK-NEXT: }
// CHECK-NEXT: }
}