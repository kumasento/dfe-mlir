#include "dfe/Dialect/MaxJ/IR/MaxJOps.h"
#include "dfe/Dialect/MaxJ/IR/MaxJDialect.h"

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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

using namespace mlir;
using namespace dfe;

// ---------------------------  MaxJ Operations

// ----------- ConstOp
static ParseResult parseConstOp(OpAsmParser &parser, OperationState &result) {
  FloatAttr val;
  Type type;

  // Note that this API will parse a value without curly brackets,
  // i.e., not a dict-type.
  // Also, the attribute should have a colon-type attached.
  //
  // parseOptionalAttrDict allows you to attach some arbitrary info,
  // but the value attribute should always be placed as <float> `:` f64
  if (parser.parseAttribute(val, "value", result.attributes) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();

  if (parser.parseArrow())
    return failure();
  if (parser.parseType(type))
    return failure();

  return parser.addTypeToList(type, result.types);
}

static void print(OpAsmPrinter &printer, maxj::ConstOp op) {
  printer << op.getOperationName() << " ";
  printer.printAttribute(op.valueAttr());
  printer << " -> " << op.getType();
}

// ----------- InputOp
static ParseResult parseInputOp(OpAsmParser &parser, OperationState &result) {
  StringAttr inputName;
  Type resultType;

  // get the name of the input exposed to the external world
  if (parser.parseAttribute(inputName, "name", result.attributes))
    return failure();

  // if another argument follows the name, it should be the enable SVar
  if (succeeded(parser.parseOptionalComma())) {
    OpAsmParser::OperandType enable;
    Type enableType;

    if (parser.parseOperand(enable))
      return failure();
    if (succeeded(parser.parseColonType(enableType))) {
      if (!enableType.isa<maxj::SVarType>() ||
          !enableType.dyn_cast<maxj::SVarType>().getType().isInteger(1))
        parser.emitError(parser.getCurrentLocation(),
                         "The 'enable' operand should be svar<i1>");
    }

    // Update the operand to the parsing result.
    // https://github.com/llvm/llvm-project/blob/a6d6b0ac93095571de743ea1f63f0b421687a275/mlir/examples/toy/Ch2/mlir/Dialect.cpp#L63
    parser.resolveOperand(enable, enableType, result.operands);
  }

  // The type of the returned value.
  if (parser.parseArrow())
    return failure();
  if (parser.parseType(resultType))
    return failure();

  return parser.addTypeToList(resultType, result.types);
}

static void print(OpAsmPrinter &printer, maxj::InputOp op) {
  printer << op.getOperationName();
  printer << " \"" << op.getAttrOfType<StringAttr>("name").getValue() << "\"";
  if (op.getNumOperands() >= 1) {
    printer << ", " << op.getOperand(0) << " : " << op.getOperand(0).getType();
  }
  printer << " -> " << op.getType() << "\n";
}

// ----------- KernelOp
static ParseResult parseKernelOp(OpAsmParser &parser, OperationState &result) {
  return success();
}

static void printArgumentList(OpAsmPrinter &printer,
                              std::vector<BlockArgument> args) {
  printer << "(";
  llvm::interleaveComma(args, printer, [&](BlockArgument arg) {
    printer << arg << " : " << arg.getType();
  });
  printer << ")";
}

static void print(OpAsmPrinter &printer, maxj::KernelOp op) {
  auto kernelName =
      op.getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()).getValue();
  printer << op.getOperationName() << " ";
  // Why we should use a specific API for printing the symbol name?
  printer.printSymbolName(kernelName);
  printer << " ";
  printArgumentList(printer, op.body().front().getArguments());
  printer << " -> ()";

  printer.printOptionalAttrDictWithKeyword(
      op.getAttrs(),
      /*elidedAttrs =*/{SymbolTable::getSymbolAttrName(),
                        maxj::KernelOp::getTypeAttrName(), "ins"});
  // what are these two false booleans stand for?
  printer.printRegion(op.body(), false, false);
}

static LogicalResult verify(maxj::KernelOp op) { return success(); }

LogicalResult maxj::KernelOp::verifyType() { return success(); }

LogicalResult maxj::KernelOp::verifyBody() { return success(); }

// why should we explicity say this?
Region *maxj::KernelOp::getCallableRegion() {
  return isExternal() ? nullptr : &getBody();
}

ArrayRef<mlir::Type> maxj::KernelOp::getCallableResults() {
  return getType().getResults();
}

#include "dfe/Dialect/MaxJ/IR/MaxJEnums.cpp.inc"
namespace dfe {
namespace maxj {
#define GET_OP_CLASSES
#include "dfe/Dialect/MaxJ/IR/MaxJ.cpp.inc"
} // namespace maxj
} // namespace dfe