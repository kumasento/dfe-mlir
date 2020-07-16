// All arithmetic operations are here.

def MaxJ_AddOp : MaxJ_SVarBinaryArithmeticOp<"add", []> {
  let summary = "Binary addition operation.";

  let description = [{
    Add up two SVar operands.

    TODO: Should extend this to support vector types later.

    Syntax:

    ```
    add-op := ssa-id `=` `maxj.add` ssa-lhs `,` ssa-rhs attr-dict `:` type
    ```

    Example:

    ```mlir
    %0 = maxj.svar : i32
    %1 = maxj.svar : i32
    %2 = maxj.add %0, %1 : !maxj.svar<i32>
    ```
  }];
}

def MaxJ_MulOp : MaxJ_SVarBinaryArithmeticOp<"mul", []> {
  let summary = "Binary multiplication operation.";

  let description = [{
    Multiply two SVar operands.

    TODO: Should extend this to support vector types later.

    Syntax:

    ```
    mul-op := ssa-id `=` `maxj.mul` ssa-lhs `,` ssa-rhs attr-dict `:` type
    ```

    Example:

    ```mlir
    %0 = maxj.svar : i32
    %1 = maxj.svar : i32
    %2 = maxj.mul %0, %1 : !maxj.svar<i32>
    ```
  }];
}