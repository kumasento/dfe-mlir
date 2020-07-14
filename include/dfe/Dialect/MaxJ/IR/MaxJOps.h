#ifndef DFE_DIALECT_MAXJ_IR_MAXJOPS_H
#define DFE_DIALECT_MAXJ_IR_MAXJOPS_H

#include "dfe/Dialect/MaxJ/IR/MaxJDialect.h"
#include "dfe/Dialect/MaxJ/IR/MaxJEnums.h.inc"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace dfe {
namespace maxj {
#define GET_OP_CLASSES
#include "dfe/Dialect/MaxJ/IR/MaxJ.h.inc"

}  // namespace maxj
}  // namespace dfe

#endif