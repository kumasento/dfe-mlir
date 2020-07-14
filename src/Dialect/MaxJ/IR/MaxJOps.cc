#include "dfe/Dialect/MaxJ/IR/MaxJOps.h"
#include "dfe/Dialect/MaxJ/IR/MaxJDialect.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace dfe;

// ---------------------------  MaxJ Operations

// ----------- ConstOp
static ParseResult parseConstOp(OpAsmParser &parser, OperationState &result) {
  Attribute val;
  Type type;

  // figure out the constant value from result attributes.
  if (parser.parseAttribute(val, "value", result.attributes) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (parser.parseOptionalColon().value ||
      !parser.parseOptionalType(type).hasValue())
    type = val.getType();

  return parser.addTypeToList(val.getType(), result.types);
}

static void print(OpAsmPrinter &printer, maxj::ConstOp op) {
  printer << op.getOperationName() << " ";
  printer.printAttributeWithoutType(op.valueAttr());
  printer.printOptionalAttrDict(op.getAttrs(), {"value"});
  printer << " : " << op.getType();
}

#include "dfe/Dialect/MaxJ/IR/MaxJEnums.cpp.inc"
namespace dfe {
namespace maxj {
#define GET_OP_CLASSES
#include "dfe/Dialect/MaxJ/IR/MaxJ.cpp.inc"
}  // namespace maxj
}  // namespace dfe