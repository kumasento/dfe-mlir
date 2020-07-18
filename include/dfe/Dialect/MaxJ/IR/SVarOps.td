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
def MaxJ_OffsetOp : MaxJ_Op<"offset", [
  NoSideEffect,
  HasParent<"KernelOp">
]> {
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


def MaxJ_ConnOp : MaxJ_Op<"conn", [
  HasParent<"KernelOp">
]> {
  let summary = "Connect two SVars.";

  let description = [{
    Connect SVar A to SVar B.

    Example:

    ```maxj
    %0 = maxj.svar : !maxj.svar<i32>
    %1 = maxj.input "a" -> !maxj.svar<i32>
    maxj.conn %1, %0 : !maxj.svar<i32>
    ```
  }];

  let arguments = (ins
    MaxJ_AnySVarType : $src,
    MaxJ_AnySVarType : $dst);
}