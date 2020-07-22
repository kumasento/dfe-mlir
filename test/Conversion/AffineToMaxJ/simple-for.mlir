// RUN: dfe-opt %s --convert-affine-to-maxj | FileCheck %s

// CHECK-LABEL: empty_body
func @empty_body () -> () {
  // CHECK: maxj.kernel @a (%arg0 : index) -> () {
  affine.for %i = 0 to 10 {
    affine.yield 
  // CHECK-NEXT: }
  }
  return 
}

// func @simple_for () -> () {
//   %0 = alloc() : memref<10xf32>
//   %1 = alloc() : memref<10xf32>
//   %2 = alloc() : memref<10xf32>
//   affine.for %i = 0 to 10 {
//     %a = affine.load %0[%i]: memref<10xf32>
//     %b = affine.load %1[%i]: memref<10xf32>
//     %c = addf %a, %b : f32
//     affine.store %c, %2[%i]: memref<10xf32>
//   }

//   return 
// }