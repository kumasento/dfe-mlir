// APIs in the IO module

// Should we add more traits?
def MaxJ_InputOp : MaxJ_Op<"input", [
  NoSideEffect, 
  HasParent<"KernelOp">
]> {
  let summary = "Create an input interface to a Kernel.";
  let description = [{
    Equivalent to an `io.input` API.


  }];

  let arguments = (ins 
    StrAttr : $name, 
    Optional<MaxJ_AnySVarType>: $enable
  );
  let results = (outs MaxJ_AnySVarType: $out);
}