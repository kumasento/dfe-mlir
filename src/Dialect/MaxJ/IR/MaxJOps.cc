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

// ----------- CounterOp
static ParseResult parseCounterOp(OpAsmParser &parser, OperationState &result) {
  IntegerAttr bitWidth, wrapPoint;
  Type type;

  if (parser.parseAttribute(bitWidth, "bitWidth", result.attributes) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // the wrapPoint attribute is optional
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseAttribute(wrapPoint, "wrapPoint", result.attributes) ||
        parser.parseOptionalAttrDict(result.attributes))
      return failure();
  }

  if (parser.parseArrow() || parser.parseType(type))
    return failure();

  return parser.addTypeToList(type, result.types);
}

static void print(OpAsmPrinter &printer, maxj::CounterOp op) {
  printer << op.getOperationName() << " " << op.getAttr("bitWidth");

  if (op.getAttr("wrapPoint"))
    printer << ", " << op.getAttr("wrapPoint");

  printer << " -> " << op.getResult().getType();
}

// ----------- SVarOp
static ParseResult parseSVarOp(OpAsmParser &parser, OperationState &result) {
  Type type;

  if (parser.parseColonType(type))
    return failure();

  auto svarType = maxj::SVarType::get(type);

  return parser.addTypeToList(svarType, result.types);
}

static void print(OpAsmPrinter &printer, maxj::SVarOp op) {
  printer << op.getOperationName() << " : " << op.getResult().getType();
}

// ----------- OffsetOp
static ParseResult parseOffsetOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType operand;
  IntegerAttr offset;
  Type type;

  // handle the offset attribute
  if (parser.parseAttribute(offset, "offset", result.attributes) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // deal with the input operand
  if (parser.parseComma() || parser.parseOperand(operand) ||
      parser.parseColonType(type))
    return failure();
  parser.resolveOperand(operand, type, result.operands);

  // the type of the result should be the same as the input
  return parser.addTypeToList(type, result.types);
}

static void print(OpAsmPrinter &printer, maxj::OffsetOp op) {
  printer << op.getOperationName() << " ";
  printer << op.offset() << " : " << op.getAttr("offset").getType();
  printer << " , " << op.getOperand() << " : " << op.getOperand().getType();
}

// ----------- SVarBinaryArithmeticOp

static ParseResult parseSVarBinaryArithmeticOp(OpAsmParser &parser,
                                               OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 2> operands;
  llvm::SMLoc operandsLoc = parser.getCurrentLocation();
  Type type;

  // parse the operands and the type
  if (parser.parseOperandList(operands, /*requiredOperandCount=*/2) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type))
    return failure();

  if (parser.resolveOperands(operands, type, result.operands))
    return failure();

  result.addTypes(type);

  return success();
}

// A generic printer for binary operations
static void printSVarBinaryArithmeticOp(OpAsmPrinter &printer, Operation *op) {
  printer << op->getName() << " " << op->getOperands();
  printer.printOptionalAttrDict(op->getAttrs());

  // The type of the single result.
  printer << " : " << op->getResult(0).getType();
}

// ----------- CastOp
static ParseResult parseCastOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType operand;
  Type operandType;
  Type resultType;

  if (parser.parseOperand(operand))
    return failure();
  if (parser.parseColonType(operandType))
    return failure();
  parser.resolveOperand(operand, operandType, result.operands);

  if (parser.parseArrow() || parser.parseType(resultType))
    return failure();

  return parser.addTypeToList(resultType, result.types);
}

static void print(OpAsmPrinter &printer, maxj::CastOp op) {
  printer << op.getOperationName() << " " << op.getOperand() << " : "
          << op.getOperand().getType() << " -> " << op.getResult().getType();
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
          !enableType.dyn_cast<maxj::SVarType>().getUnderlyingType().isInteger(
              1))
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

// ----------- OutputOp
static ParseResult parseOutputOp(OpAsmParser &parser, OperationState &result) {
  StringAttr outputName;

  // get the name of the input exposed to the external world
  if (parser.parseAttribute(outputName, "name", result.attributes))
    return failure();

  // the SVar to be output, parse it as an operand
  OpAsmParser::OperandType output;
  Type outputType;

  if (parser.parseComma())
    return failure();
  if (parser.parseOperand(output))
    return failure();
  if (parser.parseColonType(outputType))
    return failure();
  parser.resolveOperand(output, outputType, result.operands);

  // if another argument follows the name, it should be the enable SVar
  if (succeeded(parser.parseOptionalComma())) {
    OpAsmParser::OperandType enable;
    Type enableType;

    if (parser.parseOperand(enable))
      return failure();
    // check whether the SVar is a boolean
    if (succeeded(parser.parseColonType(enableType))) {
      if (!enableType.isa<maxj::SVarType>() ||
          !enableType.dyn_cast<maxj::SVarType>().getUnderlyingType().isInteger(
              1))
        parser.emitError(parser.getCurrentLocation(),
                         "The 'enable' operand should be svar<i1>");
    }

    parser.resolveOperand(enable, enableType, result.operands);
  }

  return success();
}

static void print(OpAsmPrinter &printer, maxj::OutputOp op) {
  printer << op.getOperationName();
  printer << " \"" << op.getAttrOfType<StringAttr>("name").getValue() << "\"";

  // print the output SVar
  printer << ", " << op.getOperand(0) << " : " << op.getOperand(0).getType();

  // print the optional enable SVar
  if (op.getNumOperands() >= 1) {
    printer << ", " << op.getOperand(1) << " : " << op.getOperand(0).getType();
  }
}

// ----------- MemOp
static ParseResult parseAllocOp(OpAsmParser &parser, OperationState &result) {
  IntegerAttr numElements;
  Type type;

  // should add a pair of bracket after the keyword
  // there might be arguments in it
  if (parser.parseLParen() || parser.parseRParen() ||
      parser.parseColonType(type))
    return failure();

  return parser.addTypeToList(type, result.types);
}

static void print(OpAsmPrinter &printer, maxj::AllocOp op) {
  printer << op.getOperationName() << "() : " << op.getResult().getType();
}

// ----------- KernelOp

// parse the argument list provided to a kernel.
// (%arg0 : T0, %arg1 : T1, <...>)
static ParseResult
parseArgumentList(OpAsmParser &parser,
                  SmallVectorImpl<OpAsmParser::OperandType> &args,
                  SmallVectorImpl<Type> &argTypes) {

  if (parser.parseLParen())
    return failure();

  do {
    OpAsmParser::OperandType arg;
    Type argType;

    // extract one operand from the argument list
    if (succeeded(parser.parseOptionalRegionArgument(arg))) {
      if (!arg.name.empty() && succeeded(parser.parseColonType(argType))) {
        // the parsed arg is valid.
        args.push_back(arg);
        argTypes.push_back(argType);
      }
    }
  } while (succeeded(parser.parseOptionalComma()));

  if (parser.parseRParen())
    return failure();

  return success();
}

// parse the signature for maxj.kernel
// (%arg0 : T0, %arg1 : T1, <...>) -> (%out0 : T0, %out1 : T1, <...>)
//
// Although a kernel normally won't have an output value.
static ParseResult
parseKernelSignature(OpAsmParser &parser, OperationState &result,
                     SmallVectorImpl<OpAsmParser::OperandType> &args,
                     SmallVectorImpl<Type> &argTypes) {
  if (parseArgumentList(parser, args, argTypes))
    return failure();

  // record the number of input arguments
  // again, this might not be useful.
  IntegerAttr insAttr = parser.getBuilder().getI64IntegerAttr(args.size());
  result.addAttribute("ins", insAttr);

  if (parser.parseArrow() || parseArgumentList(parser, args, argTypes))
    return failure();

  return success();
}

static ParseResult parseKernelOp(OpAsmParser &parser, OperationState &result) {
  StringAttr kernelName;
  SmallVector<OpAsmParser::OperandType, 4> args;
  SmallVector<Type, 4> argTypes;

  // get the kernel name (a symbol)
  if (parser.parseSymbolName(kernelName, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  if (parseKernelSignature(parser, result, args, argTypes))
    return failure();

  // you can attach an attribute dictionary
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  // create a type for the whole kernel.
  // from a list of arguments to none result
  auto type = parser.getBuilder().getFunctionType(argTypes, llvm::None);
  // This line wraps a type into an attribute of the kernel
  result.addAttribute(maxj::KernelOp::getTypeAttrName(), TypeAttr::get(type));

  auto *body = result.addRegion();
  // use the arguments and types from the signature to parse a region.
  parser.parseRegion(*body, args, argTypes);
  // if there is a terminator, we're good; if not, create a new one.
  maxj::KernelOp::ensureTerminator(*body, parser.getBuilder(), result.location);

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