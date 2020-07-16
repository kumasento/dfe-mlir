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


def MaxJ_CounterOp : MaxJ_Op<"counter", [
  NoSideEffect,
  HasParent<"KernelOp">
]> {
  let summary = "Create a simple counter.";

  let description = [{
    `maxj.counter` creates a SVar that acts like a counter.

    It is equivalent to the `control.count.simpleCounter` API.

    TODO: support SVar as wrap point.

    Example:

    ```mlir
    %0 = maxj.counter 1 : i64 -> !maxj.svar<i1>
    %1 = maxj.counter 8: i32, 128: i64 -> !maxj.svar<u8>
    ```
  }];

  let arguments = (ins
    I64Attr : $bitWidth,
    OptionalAttr<I64Attr>: $wrapPoint);
  let results = (outs MaxJ_AnySVarType: $out);
}