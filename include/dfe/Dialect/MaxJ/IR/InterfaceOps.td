// APIs in the IO module

// Should we add more traits?
def MaxJ_InputOp : MaxJ_Op<"input", [
  NoSideEffect, 
  HasParent<"KernelOp">
]> {
  let summary = "Create an input interface to a Kernel.";
  let description = [{
    Equivalent to an `io.input` API.

    Syntax:

    ```
    input-op := ssa-id `=` 
      `maxj.input` name (, enable-value `:` enable-type)? `->` result-type
    ```

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

def MaxJ_OutputOp : MaxJ_Op<"output", [
  NoSideEffect,
  HasParent<"KernelOp">
]> {
  let summary = "Create an output interface from a Kernel.";
  let description = [{
    Equivalent to an `io.output` API.

    Syntax:

    ```
    output-op := 
      `maxj.output` name, out-val `:` out-type, (, enable-value `:` enable-type)?
    ```

    Example:

    ```mlir
    %0 = maxj.const 1.0 : f64 -> !maxj.svar<i1>
    %1 = maxj.const 3.14 : f64 -> !maxj.svar<f64>
    maxj.output "a", %1 : !maxj.svar<f64>, %0: !maxj.svar<i1>
    ```
  }];

  let arguments = (ins 
    StrAttr : $name, 
    MaxJ_AnySVarType: $out,
    Optional<MaxJ_AnySVarType>: $enable
  );

}