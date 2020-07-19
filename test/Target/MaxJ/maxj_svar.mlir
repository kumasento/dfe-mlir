// RUN: dfe-translate --maxj-to-maxj %s | FileCheck %s

// CHECK: public class SVarInit extends Kernel {
// CHECK-NEXT: public SVarInit(KernelParameters params) {
// CHECK-NEXT: super(params);
maxj.kernel @SVarInit () -> () {
  // CHECK-NEXT: DFEVar _0 = dfeInt(1).newInstance(getOwner());
  %0 = maxj.svar : i1 
  // CHECK-NEXT: DFEVector<DFEVar> _1 = (new DFEVectorType<DFEVar>(dfeInt(8), 4)).newInstance(getOwner());
  %1 = maxj.svar : vector<4xi8>
// CHECK-NEXT: }
// CHECK-NEXT: }
}

// CHECK: public class SVarOffset extends Kernel {
// CHECK-NEXT: public SVarOffset(KernelParameters params) {
// CHECK-NEXT: super(params);
maxj.kernel @SVarOffset() -> () {
  // CHECK-NEXT: DFEVar _0 = io.input("a", dfeInt(32));
  %0 = maxj.input "a" -> !maxj.svar<i32>
  // CHECK-NEXT: DFEVar _1 = stream.offset(_0, -1);
  %1 = maxj.offset -1 : i64, %0 : !maxj.svar<i32>
  // CHECK-NEXT: DFEVector<DFEVar> _2 = io.input("vec_a", (new DFEVectorType<DFEVar>(dfeInt(32), 8)));
  %v0 = maxj.input "vec_a" -> !maxj.svar<vector<8xi32>>
  // CHECK-NEXT: DFEVector<DFEVar> _3 = stream.offset(_2, -1);
  %v1 = maxj.offset -1 : i64, %v0 : !maxj.svar<vector<8xi32>>
// CHECK-NEXT: }
// CHECK-NEXT: }
}

// CHECK: public class SVarConn extends Kernel {
// CHECK-NEXT: public SVarConn(KernelParameters params) {
// CHECK-NEXT: super(params);
maxj.kernel @SVarConn() -> () {
  // CHECK-NEXT: DFEVar _0 = dfeInt(32).newInstance(getOwner());
  %0 = maxj.svar : i32
  // CHECK-NEXT: DFEVar _1 = io.input("a", dfeInt(32));
  %1 = maxj.input "a" -> !maxj.svar<i32>
  // CHECK-NEXT: _0 <== _1;
  maxj.conn %1, %0 : !maxj.svar<i32>
// CHECK-NEXT: }
// CHECK-NEXT: }
}