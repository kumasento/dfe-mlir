def MaxJ_KernelOp : MaxJ_Op<"kernel", [
  Symbol,
  FunctionLike,
  IsolatedFromAbove,
  SingleBlockImplicitTerminator<"TerminatorOp">,
  DeclareOpInterfaceMethods<CallableOpInterface>
]> {
  let summary = "Create a MaxJ Kernel.";

  let description = [{
    `maxj.kernel` creates an entity that is equivalent to MaxJ Kernel.

    A kernel contains one region with a single block and an implicit
    `TerminatorOp` terminator. Both the block name and terminator are
    omitted in the custom syntax.

    No further blocks and control-flow are legal inside a kernel.

    You should pass a symbol attribute to a kernel.

    Kernel has no output, and its inputs are design parameters.

    Syntax:

    ```
    kernel-op ::= `maxj.kernel` kernel-symbol `(` arg-list `)` `->`
      `(` `)` attr-dict kernel-region
    ```

    Example:

    ```mlir
    ```
  }];

  let arguments = (ins Variadic<AnyInteger>: $values);
  let regions = (region SizedRegion<1>: $body);


  let extraClassDeclaration = [{
    friend class OpTrait::FunctionLike<KernelOp>;

    // use FunctionLike traits's getBody method
    using OpTrait::FunctionLike<KernelOp>::getBody;

    /// Hooks for the input/output type enumeration in FunctionLike.
    unsigned getNumFuncArguments() { return getType().getNumInputs(); }
    unsigned getNumFuncResults() { return getType().getNumResults(); }

    /// Hook for FunctionLike verifier.
    LogicalResult verifyType();

    /// Verifies the body of the function.
    LogicalResult verifyBody();
  }];

}


def MaxJ_TerminatorOp : MaxJ_Op<"terminator", [
    Terminator,
    HasParent<"KernelOp">
  ]> {
  let summary = "Dummy terminator";
  let description = [{
  }];

  let parser = ?;
  let printer = ?;
}
