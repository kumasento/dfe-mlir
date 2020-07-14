// Initialize values in MaxJ

// NOTE: using _ will create an Operation class named ConstOp instead of 
// MaxJConstOp.
def MaxJ_ConstOp : MaxJOp<"const", [ConstantLike, NoSideEffect]> {
  let summary = "Instantiate a new constant value.";

  let description = [{
    `maxj.const` returns a constant variable, which will be 
    translated to `constant.var(...)` in MaxJ.

    The value in the constant holds in the attribute.

    The result type should be `maxj.svar<...>`.

    It is an SSA operator.

    Syntax:

    ```
    const-op := ssa-id `=` 
      `maxj.const` attribute-value attr-dict `:` result-type
    ```
  }];

  let arguments = (ins AnyAttr: $value);
  let results = (outs MaxJ_AnySVarType: $out);

  // what's this?
  // allow fold values?
  let hasFolder = 0;
}