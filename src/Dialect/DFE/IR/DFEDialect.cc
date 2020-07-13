#include "dfe/Dialect/DFE/IR/DFEDialect.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace dfe;
using namespace mlir;

DFEDialect::DFEDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addOperations<
#define GET_OP_LIST
#include "dfe/Dialect/DFE/IR/DFE.cpp.inc"
      >();
}