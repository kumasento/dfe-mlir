// APIs in the IO module

// Should we add more traits?
def MaxJ_InputOp : MaxJ_Op<"input", [
  NoSideEffect, 
  HasParent<"KernelOp">
]> {
  let summary = "Create an input interface to a Kernel.";
  let description = [{
    Equivalent to an `io.input` API.

    Example:

    ```mlir
    %0 = maxj.const 1.0 : f64 -> !maxj.svar<i1>
    %1 = maxj.input "a", %0 : !maxj.svar<i1> -> !maxj.svar<f64>
    ```
  }];

  let arguments = (ins 
    StrAttr : $name, 
    Optional<MaxJ_AnySVarType>: $enable
  );
  let results = (outs MaxJ_AnySVarType: $out);
}