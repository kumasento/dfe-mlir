#ifndef DFE_TARGET_MAXJ_TRANSLATETOMAXJ_H
#define DFE_TARGET_MAXJ_TRANSLATETOMAXJ_H

namespace llvm {
class raw_ostream;
}

namespace mlir {
struct LogicalResult;
class ModuleOp;
} // namespace mlir

namespace dfe {
namespace maxj {
mlir::LogicalResult printMaxJ(mlir::ModuleOp module, llvm::raw_ostream &os);

void registerToMaxJTranslation();
} // namespace maxj
} // namespace dfe

#endif