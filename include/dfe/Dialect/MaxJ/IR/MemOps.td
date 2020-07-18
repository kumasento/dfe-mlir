// Memory related operations.

def MaxJ_MemOp : MaxJ_Op<"mem", [
  HasParent<"KernelOp">
]> {
  let summary = "Allocate a memory.";
  let description = [{
    Allocate a piece of BRAM in a kernel.
    
    Example:
    
    ```mlir
    %0 = maxj.mem 128: i64 -> !maxj.svar<i32>
    ```
  }];

  let arguments = (ins I64Attr : $numElements);
  let results = (outs MaxJ_AnyMemType : $output);
}