#include "dfe/Dialect/MaxJ/IR/MaxJDialect.h"
#include "dfe/Target/MaxJ/TranslateToMaxJ.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Translation.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;
using namespace dfe;

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::init("-"));

static llvm::cl::opt<std::string>
    outputFilename("o", llvm::cl::desc("Output filename"),
                   llvm::cl::value_desc("filename"), llvm::cl::init("-"));

static llvm::cl::opt<bool>
    splitInputFile("split-input-file",
                   llvm::cl::desc("Split the input file into pieces and "
                                  "process each chunk independently"),
                   llvm::cl::init(false));

static llvm::cl::opt<bool> verifyDiagnostics(
    "verify-diagnostics",
    llvm::cl::desc("Check that emitted diagnostics match "
                   "expected-* lines on the corresponding line"),
    llvm::cl::init(false));

int main(int argc, char **argv) {
  // Register MLIR stuff.
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  registerDialect<StandardOpsDialect>();

  // MaxJ
  registerDialect<maxj::MaxJDialect>();
  maxj::registerToMaxJTranslation();

  llvm::InitLLVM y(argc, argv);

  // Add flags for all the registered translations.
  llvm::cl::opt<const TranslateFunction *, false, TranslationParser>
      translationRequested("", llvm::cl::desc("Translation to perform"),
                           llvm::cl::Required);

  llvm::cl::ParseCommandLineOptions(argc, argv, "DFE translation driver\n");

  std::string errorMessage;
  auto input = openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  // Processes the memory buffer with a new MLIRContext.
  auto processBuffer = [&](std::unique_ptr<llvm::MemoryBuffer> ownedBuffer,
                           raw_ostream &os) {
    MLIRContext context;

    // Nothing here is threaded.  Disable synchronization overhead.
    context.disableMultithreading();

    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(ownedBuffer), llvm::SMLoc());

    if (!verifyDiagnostics) {
      SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
      return (*translationRequested)(sourceMgr, os, &context);
    }

    // In the diagnostic verification flow, we ignore whether the translation
    // failed (in most cases, it is expected to fail). Instead, we check if the
    // diagnostics were produced as expected.
    SourceMgrDiagnosticVerifierHandler sourceMgrHandler(sourceMgr, &context);
    (*translationRequested)(sourceMgr, os, &context);
    return sourceMgrHandler.verify();
  };

  if (splitInputFile) {
    if (failed(splitAndProcessBuffer(std::move(input), processBuffer,
                                     output->os())))
      return 1;
  } else {
    if (failed(processBuffer(std::move(input), output->os())))
      return 1;
  }

  output->keep();
  return 0;
}