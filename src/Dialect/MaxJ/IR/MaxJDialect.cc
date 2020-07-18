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
  addTypes<SVarType, MemType>();
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

// Shaped Type Storage.
// from mlir/lib/IR/TypeDetail.h
struct ShapedTypeStorage : public TypeStorage {
  ShapedTypeStorage(Type elementTy, unsigned subclassData = 0)
      : TypeStorage(subclassData), elementType(elementTy) {}

  /// The hash key used for uniquing.
  using KeyTy = Type;
  bool operator==(const KeyTy &key) const { return key == elementType; }

  Type elementType;
};

// Mem holds a SVar type and the shape of the memory block.
// The shape related information are partially stored in the ShapeTypeStorage
// and partially in this class.
// We follow the definition style from the VectorType definition
// in the mlir/lib/IR/TypeDetail.h
struct MemTypeStorage : public ShapedTypeStorage {
  MemTypeStorage(unsigned shapeSize, Type type, const int64_t *shapeElements)
      : ShapedTypeStorage(type, shapeSize), shapeElements(shapeElements) {}

  // The key type information is a pair here: <shape, type>
  using KeyTy = std::pair<ArrayRef<int64_t>, Type>;
  bool operator==(const KeyTy &key) const {
    return key == KeyTy(getShape(), elementType);
  }

  // Construct the MemType.
  static MemTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                   const KeyTy &key) {
    // assign the shape information
    ArrayRef<int64_t> shape = allocator.copyInto(key.first);

    return new (allocator.allocate<MemTypeStorage>())
        MemTypeStorage(shape.size(), key.second, shape.data());
  }

  ArrayRef<int64_t> getShape() const {
    return ArrayRef<int64_t>(shapeElements, getSubclassData());
  }

  const int64_t *shapeElements;
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
    parser.emitError(loc, "No underlying type found");
    return nullptr;
  }

  if (parser.parseGreater())
    return Type();

  return SVarType::get(type);
}

static Type parseMemType(DialectAsmParser &parser) {
  Type type;
  llvm::StringRef typeKeyword;

  if (parser.parseLess())
    return Type();

  // parse memory number of elements (single dimension)
  llvm::SMLoc loc = parser.getCurrentLocation();
  SmallVector<int64_t, 4> dimensions;
  if (parser.parseDimensionList(dimensions))
    return nullptr;
  if (dimensions.empty())
    return (parser.emitError(loc, "expected memory size in mem type"), nullptr);
  if (dimensions.size() > 1)
    return (parser.emitError(loc, "expected a single scalar for mem size"),
            nullptr);

  llvm::SMLoc typeLoc = parser.getCurrentLocation();
  if (parser.parseType(type)) {
    return (parser.emitError(typeLoc, "No element type found"), nullptr);
  }

  if (parser.parseGreater())
    return Type();

  return MemType::get(dimensions, type);
}

Type MaxJDialect::parseType(DialectAsmParser &parser) const {
  llvm::StringRef typeKeyword;
  // parse the type keyword first
  if (parser.parseKeyword(&typeKeyword))
    return Type();

  if (typeKeyword == SVarType::getKeyword())
    return parseSVarType(parser);
  if (typeKeyword == MemType::getKeyword())
    return parseMemType(parser);
  return Type();
}

// ------------------------ Type printing hooks
/// type ::= !maxj.svar<type>
static void printSVarType(SVarType svar, DialectAsmPrinter &printer) {
  printer << svar.getKeyword() << "<" << svar.getUnderlyingType() << ">";
}

static void printMemType(MemType mem, DialectAsmPrinter &printer) {
  printer << mem.getKeyword() << "<";
  llvm::interleave(mem.getShape(), printer, "x");
  printer << "x" << mem.getElementType() << ">";
}

void MaxJDialect::printType(mlir::Type type, DialectAsmPrinter &printer) const {
  if (SVarType svar = type.dyn_cast<SVarType>()) {
    printSVarType(svar, printer);
  } else if (MemType mem = type.dyn_cast<MemType>()) {
    printMemType(mem, printer);
  }
}

// ------------------------ MaxJ Types

FixType FixType::get(MLIRContext *context) {
  return Base::get(context, MaxJTypes::Fix);
}

// SVarType
SVarType SVarType::get(mlir::Type type) {
  return Base::get(type.getContext(), MaxJTypes::SVar, type);
}

mlir::Type SVarType::getUnderlyingType() { return getImpl()->type; }

// MemType
MemType MemType::get(ArrayRef<int64_t> shape, mlir::Type elementType) {
  return Base::get(elementType.getContext(), MaxJTypes::Mem, shape,
                   elementType);
}
Type MemType::getElementType() { return getImpl()->elementType; }
ArrayRef<int64_t> MemType::getShape() { return getImpl()->getShape(); }