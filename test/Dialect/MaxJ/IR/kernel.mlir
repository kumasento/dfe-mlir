// RUN: dfe-opt %s -mlir-print-op-generic | dfe-opt | dfe-opt | FileCheck %s
// a normal MaxJ kernel with a single constant output

// CHECK: maxj.kernel @foo () -> () {
"maxj.kernel" () ({
  ^body:
    "maxj.terminator"() {} : () -> ()
// CHECK-NEXT: }
}) {sym_name="foo", type=()->()} : () -> ()