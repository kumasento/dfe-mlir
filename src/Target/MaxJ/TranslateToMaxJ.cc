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

  LogicalResult printSVarBinaryArithmeticOp(Operation *op, StringRef opSymbol,
                                            unsigned indentAmount = 0);

  // prints things like dfeInt(32), (new DFEVectorType<DFEType>(...)), ...
  LogicalResult printSVarUnderlyingType(maxj::SVarType svar);
  // prints the type signature like DFEType, DFEVectorType<DFEVar>, etc.
  LogicalResult printSVarTypeSignature(maxj::SVarType svar);

  LogicalResult printDFEType(mlir::Type svar);

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

    // Reset the mapping since different kernels have different scopes.
    mapValueToName.clear();
    nextValueNum = 0;

    return WalkResult::advance();
  });

  return failure(result.wasInterrupted());
}

LogicalResult MaxJPrinter::printOperation(Operation *inst,
                                          unsigned indentAmount) {
  if (auto op = dyn_cast<maxj::ConstOp>(inst)) {
    out.PadToColumn(indentAmount);

    printSVarTypeSignature(op.getResult().getType().dyn_cast<maxj::SVarType>());

    out << " " << getVariableName(inst->getResult(0));
    out << " = ";
    out << "constant.var(";
    // print the DFEType
    printSVarUnderlyingType(
        op.getResult().getType().dyn_cast<maxj::SVarType>());
    out << ", ";
    // print the value
    out << op.value().convertToDouble();
    out << ");\n";
  } else if (auto op = dyn_cast<maxj::OffsetOp>(inst)) {
    out.PadToColumn(indentAmount);

    printSVarTypeSignature(op.getResult().getType().dyn_cast<maxj::SVarType>());
    out << " " << getVariableName(inst->getResult(0));
    out << " = "
        << "stream.offset(" << getVariableName(op.getOperand()) << ", "
        << op.offset() << ");\n";

  } else if (auto op = dyn_cast<maxj::SVarOp>(inst)) {
    out.PadToColumn(indentAmount);

    printSVarTypeSignature(op.getResult().getType().dyn_cast<maxj::SVarType>());

    out << " " << getVariableName(inst->getResult(0)) << " = ";
    printSVarUnderlyingType(
        op.getResult().getType().dyn_cast<maxj::SVarType>());
    out << ".newInstance(getOwner());\n";
  } else if (auto op = dyn_cast<maxj::AddOp>(inst)) {
    printSVarBinaryArithmeticOp(inst, "+", indentAmount);
  } else if (auto op = dyn_cast<maxj::MulOp>(inst)) {
    printSVarBinaryArithmeticOp(inst, "*", indentAmount);
  } else if (auto op = dyn_cast<maxj::CounterOp>(inst)) {
    out.PadToColumn(indentAmount);

    printSVarTypeSignature(op.getResult().getType().dyn_cast<maxj::SVarType>());

    out << " " << getVariableName(op.getResult()) << " = "
        << "control.count.simpleCounter(" << op.bitWidth();

    if (op.wrapPoint().hasValue()) {
      out << ", " << op.wrapPoint();
    }

    out << ");\n";
  } else if (auto op = dyn_cast<maxj::InputOp>(inst)) {
    out.PadToColumn(indentAmount);

    printSVarTypeSignature(op.getResult().getType().dyn_cast<maxj::SVarType>());

    out << " " << getVariableName(inst->getResult(0));
    out << " = "
        << "io.input(" << op.nameAttr() << ", ";
    printSVarUnderlyingType(
        op.getResult().getType().dyn_cast<maxj::SVarType>());

    if (op.getNumOperands() >= 1) {
      out << ", " << getVariableName(op.getOperand(0));
    }

    out << ")";
    out << ";\n";
  } else if (auto op = dyn_cast<maxj::OutputOp>(inst)) {
    out.PadToColumn(indentAmount);

    // NOTE: this should be changed if vector is supported.
    out << "io.output(" << op.nameAttr() << ", "
        << getVariableName(op.getOperand(0)) << ", ";
    printSVarUnderlyingType(
        op.getOperand(0).getType().dyn_cast<maxj::SVarType>());

    if (op.getNumOperands() >= 2) {
      out << ", " << getVariableName(op.getOperand(1));
    }

    out << ")";
    out << ";\n";
  } else if (auto op = dyn_cast<maxj::AllocOp>(inst)) {
    out.PadToColumn(indentAmount);

    auto memTy = op.getResult().getType().dyn_cast<maxj::MemType>();

    // Looks a bit cluttered.
    out << "Memory<";
    printSVarTypeSignature(maxj::SVarType::get(memTy.getElementType()));
    out << ">";

    out << " " << getVariableName(inst->getResult(0));
    out << " = "
        << "mem.alloc(";

    printDFEType(memTy.getElementType());
    out << ", " << memTy.getShape()[0] << ");\n";

  } else if (auto op = dyn_cast<maxj::ReadOp>(inst)) {
    out.PadToColumn(indentAmount);

    printSVarTypeSignature(op.getResult().getType().dyn_cast<maxj::SVarType>());

    out << " " << getVariableName(inst->getResult(0));
    out << " = " << getVariableName(op.getOperand(0)) << ".read(";
    out << getVariableName(op.getOperand(1));
    out << ");\n";

  } else if (auto op = dyn_cast<maxj::WriteOp>(inst)) {
    out.PadToColumn(indentAmount);

    out << getVariableName(op.getOperand(0)) << ".write(";
    out << getVariableName(op.getOperand(1)) << ", ";
    out << getVariableName(op.getOperand(2));

    if (op.getNumOperands() > 3) {
      out << ", " << getVariableName(op.getOperand(3));
    }

    out << ");\n";

  } else if (auto op = dyn_cast<maxj::ConnOp>(inst)) {
    out.PadToColumn(indentAmount);

    out << getVariableName(op.getOperand(1))
        << " <== " << getVariableName(op.getOperand(0)) << ";\n";
  }
}

Twine MaxJPrinter::getVariableName(Value value) {
  if (!mapValueToName.count(value)) {
    mapValueToName.insert(std::make_pair(value, nextValueNum));
    nextValueNum++;
  }
  return Twine("_") + Twine(mapValueToName.lookup(value));
}

LogicalResult MaxJPrinter::printSVarBinaryArithmeticOp(Operation *inst,
                                                       StringRef opSymbol,
                                                       unsigned indentAmount) {
  // sanity check
  if (inst->getNumOperands() != 2) {
    return emitError(inst->getLoc(),
                     "This operation should have two operands.");
  }
  if (inst->getNumResults() != 1) {
    return emitError(inst->getLoc(), "This operation should have one result.");
  }

  // print operation
  out.PadToColumn(indentAmount);

  printSVarTypeSignature(
      inst->getResult(0).getType().dyn_cast<maxj::SVarType>());

  out << " " << getVariableName(inst->getResult(0)) << " = "
      << getVariableName(inst->getOperand(0)) << " " << opSymbol << " "
      << getVariableName(inst->getOperand(1)) << ";\n";

  return success();
}

LogicalResult MaxJPrinter::printDFEType(mlir::Type type) {
  if (type.isIntOrFloat()) {
    // for the scalar case
    unsigned w = type.getIntOrFloatBitWidth();
    if (type.isInteger(w)) {
      out << "dfeInt(" << w << ")";
    } else if (type.isF64()) {
      out << "dfeFloat(11, 52)";
    } else if (type.isF32()) {
      out << "dfeFloat(8, 23)";
    }
    return success();
  } else if (type.isa<mlir::VectorType>()) {
    // for the vector case
    auto vectorType = type.dyn_cast<mlir::VectorType>();

    out << "(new DFEVectorType<DFEVar>(";
    // resolve the DFEType construction
    printDFEType(vectorType.getElementType());
    out << ", " << vectorType.getNumElements() << "))";
  }

  return failure();
}

// This will convert the underlying type wrapped in a SVar
// to a DFEType in MaxJ.
LogicalResult MaxJPrinter::printSVarUnderlyingType(maxj::SVarType svar) {
  mlir::Type type = svar.getUnderlyingType();
  printDFEType(type);
  return success();
}

LogicalResult MaxJPrinter::printSVarTypeSignature(maxj::SVarType svar) {
  mlir::Type type = svar.getUnderlyingType();
  // TODO: needs more sanity checks here
  if (auto ty = type.dyn_cast<mlir::VectorType>()) {
    out << "DFEVector<DFEVar>";
  } else {
    out << "DFEVar";
  }
  return success();
}

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