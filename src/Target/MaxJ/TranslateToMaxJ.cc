#include "dfe/Target/MaxJ/TranslateToMaxJ.h"
#include "dfe/Dialect/MaxJ/IR/MaxJOps.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Translation.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FormattedStream.h"

using namespace mlir;
using namespace dfe;

namespace {

class MaxJPrinter {
public:
  MaxJPrinter(llvm::formatted_raw_ostream &output) : out(output) {}

  LogicalResult printModule(ModuleOp op);
  LogicalResult printOperation(Operation *op, unsigned indentAmount = 0);

private:
  Twine getVariableName(Value value);

  LogicalResult printSVarUnderlyingType(maxj::SVarType svar);

  llvm::formatted_raw_ostream &out;
  unsigned nextValueNum = 0;
  DenseMap<Value, unsigned> mapValueToName;
  unsigned instCount = 0;
};

LogicalResult MaxJPrinter::printModule(mlir::ModuleOp module) {
  WalkResult result = module.walk([this](maxj::KernelOp kernel) -> WalkResult {
    // the single block of a kernel op.
    Block &entryBlock = kernel.body().front();

    out << "public class " << kernel.getName() << " extends Kernel {\n";
    out.PadToColumn(2);
    out << "public " << kernel.getName() << "(KernelParameters params) {\n";
    out.PadToColumn(4);
    out << "super(params);\n";
    out << "\n";

    // go through every operation in the block
    for (auto iter = entryBlock.begin();
         iter != entryBlock.end() && !dyn_cast<maxj::TerminatorOp>(iter);
         ++iter) {
      if (failed(printOperation(&(*iter), 4))) {
        return emitError(iter->getLoc(), "Operation not supported!");
      }
    }

    out.PadToColumn(2);
    out << "}\n";

    out << "}\n";

    return WalkResult::advance();
  });

  return failure(result.wasInterrupted());
}

LogicalResult MaxJPrinter::printOperation(Operation *inst,
                                          unsigned indentAmount) {
  if (auto op = dyn_cast<maxj::ConstOp>(inst)) {
    out.PadToColumn(indentAmount);
    out << "DFEVar " << getVariableName(inst->getResult(0));
    out << " = ";
    out << "constant.var(";
    // print the DFEType
    printSVarUnderlyingType(
        op.getResult().getType().dyn_cast<maxj::SVarType>());
    out << ", ";
    // print the value
    out << op.value().convertToDouble();
    out << ");\n";
  }
}

Twine MaxJPrinter::getVariableName(Value value) {
  if (!mapValueToName.count(value)) {
    mapValueToName.insert(std::make_pair(value, nextValueNum));
    nextValueNum++;
  }
  return Twine("_") + Twine(mapValueToName.lookup(value));
}

LogicalResult MaxJPrinter::printSVarUnderlyingType(maxj::SVarType svar) {
  mlir::Type type = svar.getType();
  if (type.isIntOrFloat()) {
    unsigned w = type.getIntOrFloatBitWidth();
    if (type.isInteger(w)) {
      out << "dfeInt(" << w << ")";
    }
  }
  return success();
} // namespace

} // namespace

namespace dfe {
namespace maxj {
LogicalResult printMaxJ(ModuleOp module, raw_ostream &os) {
  llvm::formatted_raw_ostream out(os);
  MaxJPrinter printer(out);
  return printer.printModule(module);
}

void registerToMaxJTranslation() {
  TranslateFromMLIRRegistration registration(
      "maxj-to-maxj", [](ModuleOp module, raw_ostream &output) {
        return printMaxJ(module, output);
      });
}

} // namespace maxj
} // namespace dfe