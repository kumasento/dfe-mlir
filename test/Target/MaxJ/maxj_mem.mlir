// RUN: dfe-translate --maxj-to-maxj %s | FileCheck %s

// CHECK: public class ScalarMemRead extends Kernel {
// CHECK-NEXT: public ScalarMemRead(KernelParameters params) {
// CHECK-NEXT: super(params);
maxj.kernel @ScalarMemRead() -> () {
  // CHECK-NEXT: Memory<DFEVar> _0 = mem.alloc(dfeInt(32), 128);
  %0 = maxj.alloc() : !maxj.mem<128xi32>
  // CHECK-NEXT: DFEVar _1 = control.count.simpleCounter(8);
  %1 = maxj.counter 8: i64 -> !maxj.svar<i8>
  // CHECK-NEXT: DFEVar _2 = _0.read(_1);
  %2 = maxj.read %0 : !maxj.mem<128xi32>, %1: !maxj.svar<i8>
// CHECK-NEXT: }
// CHECK-NEXT: }
}

// CHECK: public class ScalarMemWrite extends Kernel {
// CHECK-NEXT: public ScalarMemWrite(KernelParameters params) {
// CHECK-NEXT: super(params);
maxj.kernel @ScalarMemWrite() -> () {
  // CHECK-NEXT: Memory<DFEVar> _0 = mem.alloc(dfeInt(32), 128);
  %0 = maxj.alloc() : !maxj.mem<128xi32>
  // CHECK-NEXT: DFEVar _1 = control.count.simpleCounter(8);
  %1 = maxj.counter 8 : i64 -> !maxj.svar<i8>
  // CHECK-NEXT: DFEVar _2 = io.input("val", dfeInt(32));
  %2 = maxj.input "val" -> !maxj.svar<i32>
  // CHECK-NEXT: DFEVar _3 = io.input("enable", dfeInt(1));
  %3 = maxj.input "enable" -> !maxj.svar<i1>
  // CHECK-NEXT: _0.write(_1, _2, _3);
  maxj.write %0 : !maxj.mem<128xi32>, %1 : !maxj.svar<i8>, %2 : !maxj.svar<i32>, %3 : !maxj.svar<i1>
// CHECK-NEXT: }
// CHECK-NEXT: }
}

// CHECK: public class VectorMemRead extends Kernel {
// CHECK-NEXT: public VectorMemRead(KernelParameters params) {
// CHECK-NEXT: super(params);
maxj.kernel @VectorMemRead() -> () {
  // CHECK-NEXT: Memory<DFEVector<DFEVar>> _0 = mem.alloc((new DFEVectorType<DFEVar>(dfeInt(32), 12)), 128);
  %0 = maxj.alloc() : !maxj.mem<128xvector<12xi32>>
  // CHECK-NEXT: DFEVar _1 = control.count.simpleCounter(8);
  %1 = maxj.counter 8: i64 -> !maxj.svar<i8>
  // CHECK-NEXT: DFEVector<DFEVar> _2 = _0.read(_1);
  %2 = maxj.read %0 : !maxj.mem<128xvector<12xi32>>, %1: !maxj.svar<i8>
// CHECK-NEXT: }
// CHECK-NEXT: }
}

// CHECK: public class VectorMemWrite extends Kernel {
// CHECK-NEXT: public VectorMemWrite(KernelParameters params) {
// CHECK-NEXT: super(params);
maxj.kernel @VectorMemWrite() -> () {
  // CHECK-NEXT: Memory<DFEVector<DFEVar>> _0 = mem.alloc((new DFEVectorType<DFEVar>(dfeInt(32), 12)), 128);
  %0 = maxj.alloc() : !maxj.mem<128xvector<12xi32>>
  // CHECK-NEXT: DFEVar _1 = control.count.simpleCounter(8);
  %1 = maxj.counter 8 : i64 -> !maxj.svar<i8>
  // CHECK-NEXT: DFEVector<DFEVar> _2 = io.input("val", (new DFEVectorType<DFEVar>(dfeInt(32), 12)));
  %2 = maxj.input "val" -> !maxj.svar<vector<12xi32>>
  // CHECK-NEXT: DFEVar _3 = io.input("enable", dfeInt(1));
  %3 = maxj.input "enable" -> !maxj.svar<i1>
  // CHECK-NEXT: _0.write(_1, _2, _3);
  maxj.write %0 : !maxj.mem<128xvector<12xi32>>, %1 : !maxj.svar<i8>, %2 : !maxj.svar<vector<12xi32>>, %3 : !maxj.svar<i1>
// CHECK-NEXT: }
// CHECK-NEXT: }
}