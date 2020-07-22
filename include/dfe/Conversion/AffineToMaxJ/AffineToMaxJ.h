#ifndef DFE_CONVERSION_AFFINETOMAXJ_AFFINETOMAXJ_H
#define DFE_CONVERSION_AFFINETOMAXJ_AFFINETOMAXJ_H

#include "mlir/Support/LLVM.h"
namespace mlir {
class AffineExpr;
class AffineForOp;
class AffineMap;
class Location;
struct LogicalResult;
class MLIRContext;
class OpBuilder;
class RewritePattern;
class Value;
class ValueRange;

// Owning list of rewriting patterns.
class OwningRewritePatternList;
class FuncOp;
template <typename T>
class OperationPass;
} // namespace mlir

namespace dfe {

void populateAffineToMaxJConversionPatterns(
    mlir::OwningRewritePatternList &patterns, mlir::MLIRContext *context);

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>>
createConvertAffineToMaxJPass();
void initAffineToMaxJPasses();
} // namespace dfe

#endif