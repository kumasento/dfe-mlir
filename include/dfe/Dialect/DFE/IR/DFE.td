#ifndef DFE_DIALECT_DFE_IR_H
#define DFE_DIALECT_DFE_IR_H

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

// ---------------------- DFE Dialect definition

def DFEDialect : Dialect {
  let name = "dfe";

  let description = [{
    A streaming hardware architecture description dialect in MLIR.
  }];

  let cppNamespace = "dfe";

  // What's this?
  let hasConstantMaterializer = 1;
}

class DFEOp<string mnemonic, list<OpTrait> traits = []> :
  Op<DFEDialect, mnemonic, traits>;

// ---------------------- DFE Operations

include "ValueOps.td"

#endif // DFE_DIALECT_DFE_IR_H 
