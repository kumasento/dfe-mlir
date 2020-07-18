// SVar related operations

def MaxJ_SVarOp : MaxJ_Op<"svar", [ HasParent<"KernelOp"> ]> {
  let summary = "Create a new SVar object.";

  let description = [{
    Create DFEVar.

    Example:

    ```mlir
    %0 = maxj.svar : i64
    ```
  }];

  let results = (outs MaxJ_AnySVarType : $result);

  // NOTE: can we use the assemblyFormat to parse this svar op?
}

// ------------------ Stream related

// TODO: is there any other trait for this Op?
def MaxJ_OffsetOp : MaxJ_Op<"offset", [ HasParent<"KernelOp"> ]> {
  let summary = "Create a stream offset on an SVar object";

  let description = [{
    Get the offset element in the stream that an SVar creates.

    Example:

    ```mlir
    %0 = maxj.input "a" -> !maxj.svar<i64>
    %1 = maxj.offset -1 : i64, %0 : !maxj.svar<i64>
    ```
  }];

  let arguments = (ins
    I64Attr : $offset,
    MaxJ_AnySVarType : $input);
  let results = (outs MaxJ_AnySVarType : $output);
}