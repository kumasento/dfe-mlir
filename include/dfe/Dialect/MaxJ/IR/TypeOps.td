// manipulating the type 

def MaxJ_CastOp : MaxJ_Op<"cast", [
  NoSideEffect,
  HasParent<"KernelOp">
]> {
  let summary = "Typecast operation";
  let description = [{
    Typecast from one type to another.

    Example:

    ```mlir
    %0 = maxj.const 1 : f64 -> !maxj.svar<f64>
    %1 = maxj.cast %0 : !maxj.svar<f64> -> !maxj.svar<i32>
    ```
  }];

  let arguments = (ins MaxJ_AnySVarType: $input);
  let results = (outs MaxJ_AnySVarType: $output);
}