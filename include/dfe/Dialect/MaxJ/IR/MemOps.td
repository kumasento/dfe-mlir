// Memory related operations.

// TODO: figure out the traits
def MaxJ_AllocOp : MaxJ_Op<"alloc", [
  HasParent<"KernelOp">
]> {
  let summary = "Allocate a memory.";
  let description = [{
    Allocate a piece of BRAM in a kernel.
    
    Example:
    
    ```mlir
    %0 = maxj.alloc() : !maxj.mem<128xi32>
    ```
  }];

  let results = (outs MaxJ_AnyMemType : $output);
}