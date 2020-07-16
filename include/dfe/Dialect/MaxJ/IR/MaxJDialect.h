#ifndef DFE_DIALECT_MAXJ_IR_MAXJDIALECT_H
#define DFE_DIALECT_MAXJ_IR_MAXJDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"

using namespace mlir;

namespace dfe {

namespace maxj {

class MaxJDialect : public Dialect {
public:
  explicit MaxJDialect(MLIRContext *context);

  // the namespace prefix
  static StringRef getDialectNamespace() { return "maxj"; }

  /// Parses a type registered to this dialect
  mlir::Type parseType(DialectAsmParser &parser) const override;
  /// Print a type registered to this dialect
  void printType(mlir::Type type, DialectAsmPrinter &printer) const override;
};

// ----------------------- MaxJ Types
namespace MaxJTypes {
enum Kinds {
  Fix = mlir::Type::FIRST_PRIVATE_EXPERIMENTAL_0_TYPE,
  SVar,
};

}

// ----------------------- Fixed Point Types

// NOTE: this is not completed yet. Should make it parametric.
class FixType
    : public mlir::Type::TypeBase<FixType, mlir::Type, DefaultTypeStorage> {
public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == MaxJTypes::Fix; }

  static FixType get(MLIRContext *context);

  static llvm::StringRef getKeyword() { return "fix"; }
};

// ----------------------- SVar Type
namespace detail {
struct SVarTypeStorage;
}

class SVarType : public mlir::Type::TypeBase<SVarType, mlir::Type,
                                             detail::SVarTypeStorage> {
public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == MaxJTypes::SVar; }

  static SVarType get(mlir::Type type);
  mlir::Type getUnderlyingType();

  static llvm::StringRef getKeyword() { return "svar"; }
};

} // namespace maxj

} // namespace dfe

#endif