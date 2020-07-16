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
include "Types.td"


// ---------------------- MaxJ Operations

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


include "SVarOps.td"
include "TypeOps.td"
include "ArithmeticOps.td"
include "ValueOps.td"
include "StructureOps.td"
include "InterfaceOps.td"

#endif // DFE_DIALECT_MAXJ_IR_H 
