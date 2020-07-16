// Initialize values in MaxJ

def MaxJ_ConstOp : MaxJ_Op<"const", [
  ConstantLike,
  HasParent<"KernelOp">,
  NoSideEffect
]> {
  let summary = "Instantiate a new constant value.";

  let description = [{
    `maxj.const` returns a constant variable, which will be 
    translated to `constant.var(...)` in MaxJ.

    The value in the constant holds in the attribute. It should be a double type 
    value (for now).

    The result type should be `maxj.svar<...>`.

    Syntax:

    ```
    const-op := ssa-id `=` 
      `maxj.const` attribute-value `:` attr-type `->` result-type
    ```

    This syntax is a bit cluttered. Should update this later.
    See the MaxJOps.cc for more information.

    Example:

    ```mlir
    %0 = maxj.const 1.0 : f64 -> !maxj.svar<i64>
    ```

    Here the type of the constant value can be different from the
    type in the returned SVar type.
  }];

  let arguments = (ins F64Attr: $value);
  let results = (outs MaxJ_AnySVarType: $out);

  // what's this?
  // allow fold values?
  let hasFolder = 0;
}