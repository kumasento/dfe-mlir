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
    // memory with 128 elements and a scalar port
    %0 = maxj.alloc() : !maxj.mem<128 x i32>
    // memory with 256 elements and an 8-elements port
    %1 = maxj.alloc() : !maxj.mem<256 x vector<8xi32>>
    ```
  }];

  let results = (outs MaxJ_AnyMemType : $output);
}

// ReadOp
def MaxJ_ReadOp : MaxJ_Op<"read", [
  HasParent<"KernelOp">
]> {
  let summary = "Read from a memory block.";

  let description = [{
    `maxj.read` reads from a memory block and returns an SVar.

    Example:

    ```mlir
    %0 = maxj.alloc : !maxj.mem<128xi32>
    %1 = maxj.counter 8 : i64 -> !maxj.svar<i8>
    %2 = maxj.read %0 : !maxj.mem<128xi32>, %1 : !maxj.svar<i8>
    ```
  }];

  let arguments = (ins
    MaxJ_AnyMemType : $mem,
    MaxJ_AnySVarType : $addr);
  let results = (outs MaxJ_AnySVarType : $out);
}


// WriteOp
def MaxJ_WriteOp : MaxJ_Op<"write", [
  HasParent<"KernelOp">
]> {
  let summary = "Write value into a memory block.";

  let description = [{
    `maxj.write` writes a single SVar into a memory block.

    Example:

    ```mlir
    %0 = maxj.alloc : !maxj.mem<128xi32>
    %1 = maxj.counter 8 : i64 -> !maxj.svar<i8>
    %2 = maxj.input "val" : !maxj.svar<i32>;
    %3 = maxj.input "enable" : !maxj.svar<i1>;
    maxj.write %0 : !maxj.mem<128xi32>, %1 : !maxj.svar<i8>, %2 : !maxj.svar<i32>, %3 : !maxj.svar<i1>
    ```
  }];

  let arguments = (ins 
    MaxJ_AnyMemType : $mem,
    MaxJ_AnySVarType : $addr,
    MaxJ_AnySVarType : $data,
    Optional<MaxJ_AnySVarType> : $enable
  );
}