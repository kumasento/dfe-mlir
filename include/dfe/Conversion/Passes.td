#ifndef DFE_CONVERSION_PASSES
#define DFE_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"


def ConvertAffineToMaxJ : FunctionPass<"convert-affine-to-maxj"> {
  let summary = "Lower Affine operations to MaxJ ";
  let description = [{

  }];
  let constructor = "dfe::createConvertAffineToMaxJPass()";
}


#endif // DFE_CONVERSION_PASSES
