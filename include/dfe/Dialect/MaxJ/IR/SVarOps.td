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