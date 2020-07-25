#ifndef DFE_DIALECT_MAXJ_IR_H
#define DFE_DIALECT_MAXJ_IR_H

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

// ---------------------- MaxJ Dialect definition

def MaxJ_Dialect : Dialect {
  let name = "maxj";

  let description = [{
    A dialect for MaxJ in MLIR.
  }];

  let cppNamespace = "dfe::maxj";

  // What's this?
  let hasConstantMaterializer = 0;
}

// ---------------------- MaxJ Types
// class ContainerType<Type etype, Pred containerPred, code elementTypeCall,
//                     string descr>
// class BuildableType<code builder> {
//   // The builder call to invoke (if specified) to construct the BuildableType.
//   code builderCall = builder;
// }

/// Fixed point types

def AnyFix : Type<CPred<"$_self.isa<FixType>()">, "MaxJ Fixed Point Type">,
             BuildableType<"FixType::get($_builder.getContext())">;

/// SVar

class MaxJ_SVarType<list<Type> allowedTypes>
  : ContainerType<AnyTypeOf<allowedTypes>, CPred<"$_self.isa<SVarType>()">,
                  "$_self.cast<SVarType>().getUnderlyingType()", "MaxJ SVar type">;

// TODO: will support Tensor in the future.
def MaxJ_AnySVarAllowedType : AnyTypeOf<[AnyInteger, AnyFloat, AnyVector]>;

def MaxJ_AnySVarType : MaxJ_SVarType<[MaxJ_AnySVarAllowedType]>;

/// Memory

def MaxJ_IsMemTypePred : CPred<"$_self.isa<MemType>()">;

class MaxJ_MemOf<list<Type> allowedTypes>
  : ContainerType<AnyTypeOf<allowedTypes>, MaxJ_IsMemTypePred,
                  "$_self.cast<MemType>().getElementType()", "MaxJ Mem type">;

def MaxJ_AnyMemType : MaxJ_MemOf<[MaxJ_AnySVarAllowedType]>;


// ---------------------------------------------- MaxJ Operations

class MaxJ_Op<string mnemonic, list<OpTrait> traits = []> :
  Op<MaxJ_Dialect, mnemonic, traits> {
  // For each MaxJ op, the following static functions need to be defined in
  // MaxJOps.cc:
  //
  // * static ParseResult parse<op-c++-class-name>(OpAsmParser &parser,
  //                                               OperationState &state);
  // * static void print(OpAsmPrinter &p, <op-c++-class-name> op)
  let parser = [{ return ::parse$cppClass(parser, result); }];
  let printer = [{ ::print(p, *this); }];
}


// This type class corresponds to those binary arithmetic operators 
// works between two SVar operands of the same type.
class MaxJ_SVarBinaryArithmeticOp <string mnemonic, list<OpTrait> traits = []> :
  Op<MaxJ_Dialect, mnemonic,
     !listconcat(traits, [NoSideEffect, HasParent<"KernelOp">])> {
    
  let arguments = (ins MaxJ_AnySVarType: $lhs, MaxJ_AnySVarType: $rhs);
  let results = (outs MaxJ_AnySVarType: $out);

  let parser = [{ return ::parseSVarBinaryArithmeticOp(parser, result); }];

  let printer = [{ ::printSVarBinaryArithmeticOp(p, this->getOperation()); }];
}

// ----------------------------------------------SVarOps
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

// ----------------------------------------- TypeOps
// manipulating the type 

def MaxJ_CastOp : MaxJ_Op<"cast", [
  NoSideEffect,
  HasParent<"KernelOp">
]> {
  let summary = "Typecast operation";
  let description = [{
    Typecast from one type to another.

    Example:

    ```mlir
    %0 = maxj.const 1 : f64 -> !maxj.svar<f64>
    %1 = maxj.cast %0 : !maxj.svar<f64> -> !maxj.svar<i32>
    ```
  }];

  let arguments = (ins MaxJ_AnySVarType: $input);
  let results = (outs MaxJ_AnySVarType: $output);
}


// ----------------------------------------- MemOps
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

// ----------------------------------------- ArithmeticOps
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

// ----------------------------------------- ValueOps
// Initialize values in MaxJ

def MaxJ_ConstOp : MaxJ_Op<"const", [
  HasParent<"KernelOp">,
  NoSideEffect
]> {
  let summary = "Instantiate a new constant value.";

  let description = [{
    `maxj.const` returns a constant variable, which will be 
    translated to `constant.var(...)` in MaxJ.

    The result type should be `maxj.svar<...>`.

    Syntax:

    ```
    const-op := ssa-id `=` 
      `maxj.const` val `:` result-type
    ```

    Example:

    ```mlir
    %c0 = constant 1 : i64
    %0 = maxj.const %c0 : !maxj.svar<i64>
    ```
  }];

  let arguments = (ins AnyType: $value);
  let results = (outs MaxJ_AnySVarType: $out);
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

// ----------------------------------------- Structure Ops
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


// ----------------------------------------- Interface Ops
// Should we add more traits?
def MaxJ_InputOp : MaxJ_Op<"input", [
  NoSideEffect, 
  HasParent<"KernelOp">
]> {
  let summary = "Create an input interface to a Kernel.";
  let description = [{
    Equivalent to an `io.input` API.

    Syntax:

    ```
    input-op := ssa-id `=` 
      `maxj.input` name (, enable-value `:` enable-type)? `->` result-type
    ```

    Example:

    ```mlir
    %0 = maxj.const 1.0 : f64 -> !maxj.svar<i1>
    %1 = maxj.input "a", %0 : !maxj.svar<i1> -> !maxj.svar<f64>
    ```
  }];

  let arguments = (ins 
    StrAttr : $name, 
    Optional<MaxJ_AnySVarType>: $enable
  );
  let results = (outs MaxJ_AnySVarType: $out);
}

def MaxJ_OutputOp : MaxJ_Op<"output", [
  NoSideEffect,
  HasParent<"KernelOp">
]> {
  let summary = "Create an output interface from a Kernel.";
  let description = [{
    Equivalent to an `io.output` API.

    Syntax:

    ```
    output-op := 
      `maxj.output` name, out-val `:` out-type, (, enable-value `:` enable-type)?
    ```

    Example:

    ```mlir
    %0 = maxj.const 1.0 : f64 -> !maxj.svar<i1>
    %1 = maxj.const 3.14 : f64 -> !maxj.svar<f64>
    maxj.output "a", %1 : !maxj.svar<f64>, %0: !maxj.svar<i1>
    ```
  }];

  let arguments = (ins 
    StrAttr : $name, 
    MaxJ_AnySVarType: $out,
    Optional<MaxJ_AnySVarType>: $enable
  );

}

#endif // DFE_DIALECT_MAXJ_IR_H 
