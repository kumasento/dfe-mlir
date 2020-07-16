#include "dfe/Dialect/MaxJ/IR/MaxJDialect.h"
#include "dfe/Dialect/MaxJ/IR/MaxJOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

using namespace dfe;
using namespace dfe::maxj;
using namespace mlir;

MaxJDialect::MaxJDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addTypes<SVarType>();
  addOperations<
#define GET_OP_LIST
#include "dfe/Dialect/MaxJ/IR/MaxJ.cpp.inc"
      >();
}

namespace dfe {
namespace maxj {
namespace detail {

// SVar holds a type
struct SVarTypeStorage : public mlir::TypeStorage {
  using KeyTy = mlir::Type;

  SVarTypeStorage(mlir::Type type) : type(type) {}

  bool operator==(const KeyTy &key) const { return key == type; }

  mlir::Type type;

  /// construction method for creating a new instance of the svar type
  /// storage
  static SVarTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                    const KeyTy &key) {
    return new (allocator.allocate<SVarTypeStorage>()) SVarTypeStorage(key);
  }
};

} // namespace detail
} // namespace maxj
} // namespace dfe

// ------------------------ Type parsing

static Type parseSVarType(DialectAsmParser &parser) {
  Type type;

  if (parser.parseLess())
    return Type();

  llvm::SMLoc loc = parser.getCurrentLocation();
  if (parser.parseType(type)) {
    parser.emitError(loc, "No signal type found. Signal needs an underlying "
                          "type.");
    return nullptr;
  }

  if (parser.parseGreater())
    return Type();

  return SVarType::get(type);
}

Type MaxJDialect::parseType(DialectAsmParser &parser) const {
  llvm::StringRef typeKeyword;
  // parse the type keyword first
  if (parser.parseKeyword(&typeKeyword))
    return Type();
  if (typeKeyword == SVarType::getKeyword())
    return parseSVarType(parser);
  return Type();
}

// ------------------------ Type printing hooks
/// type ::= !maxj.svar<type>
static void printSVarType(SVarType svar, DialectAsmPrinter &printer) {
  printer << svar.getKeyword() << "<" << svar.getUnderlyingType() << ">";
}

void MaxJDialect::printType(mlir::Type type, DialectAsmPrinter &printer) const {
  if (SVarType svar = type.dyn_cast<SVarType>()) {
    printSVarType(svar, printer);
  }
}

// ------------------------ MaxJ Types

FixType FixType::get(MLIRContext *context) {
  return Base::get(context, MaxJTypes::Fix);
}

SVarType SVarType::get(mlir::Type type) {
  return Base::get(type.getContext(), MaxJTypes::SVar, type);
}

mlir::Type SVarType::getUnderlyingType() { return getImpl()->type; }