#include "dfe/Conversion/AffineToMaxJ/AffineToMaxJ.h"
#include "dfe/Dialect/MaxJ/IR/MaxJDialect.h"
#include "dfe/Dialect/MaxJ/IR/MaxJOps.h"

#include "../PassDetail.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace dfe;

namespace {

class AffineForLowering : public OpRewritePattern<AffineForOp> {
public:
  using OpRewritePattern<AffineForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineForOp op,
                                PatternRewriter &rewriter) const override {
    // Location loc = op.getLoc();
    // Value lowerBound = lowerAffineLowerBound(op, rewriter);
    // Value upperBound = lowerAffineUpperBound(op, rewriter);
    // Value step = rewriter.create<ConstantIndexOp>(loc, op.getStep());
    // auto f = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);
    // rewriter.eraseBlock(f.getBody());
    // rewriter.inlineRegionBefore(op.region(), f.region(), f.region().end());

    Location loc = op.getLoc();

    SmallVector<mlir::Type, 4> resultTypes;
    SmallVector<mlir::Value, 4> operands;
    SmallVector<mlir::NamedAttribute, 4> attributes;
    auto sym_name =
        rewriter.getNamedAttr("sym_name", rewriter.getStringAttr("a"));
    auto type_attr = rewriter.getNamedAttr(
        "type",
        mlir::TypeAttr::get(rewriter.getFunctionType(llvm::None, llvm::None)));
    attributes.push_back(sym_name);
    attributes.push_back(type_attr);
    auto kernel =
        rewriter.create<maxj::KernelOp>(loc, resultTypes, operands, attributes);

    rewriter.inlineRegionBefore(op.region(), kernel.body(),
                                kernel.body().end());
    rewriter.eraseOp(op);
    return success();
  }
};

/// Affine terminators are removed.
class AffineYieldOpLowering : public OpRewritePattern<AffineYieldOp> {
public:
  using OpRewritePattern<AffineYieldOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineYieldOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<maxj::TerminatorOp>(op);
    return success();
  }
};
} // namespace

void dfe::populateAffineToMaxJConversionPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context) {
  patterns.insert<AffineForLowering, AffineYieldOpLowering>(context);
}

namespace {
class ConvertAffineToMaxJPass
    : public dfe::ConvertAffineToMaxJBase<ConvertAffineToMaxJPass> {

public:
  // entry function to the pass
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    populateAffineToMaxJConversionPatterns(patterns, &getContext());

    ConversionTarget target(getContext());
    target.addLegalDialect<maxj::MaxJDialect>();

    if (failed(applyPartialConversion(getFunction(), target, patterns)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>>
dfe::createConvertAffineToMaxJPass() {
  return std::make_unique<ConvertAffineToMaxJPass>();
}

void dfe::initAffineToMaxJPasses() {
#define GEN_PASS_REGISTRATION
#include "dfe/Conversion/Passes.h.inc"
}