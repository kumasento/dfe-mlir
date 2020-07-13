#ifndef DFE_DIALECT_DFE_IR_DFEDIALECT_H
#define DFE_DIALECT_DFE_IR_DFEDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"

using namespace mlir;

namespace dfe {

class DFEDialect : public Dialect {
 public:
  explicit DFEDialect(MLIRContext *context);

  // the namespace prefix
  static StringRef getDialectNamespace() { return "dfe"; }
};

}  // namespace dfe

#endif