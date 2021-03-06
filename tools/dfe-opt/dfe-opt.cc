#include "dfe/Conversion/AffineToMaxJ/AffineToMaxJ.h"
#include "dfe/Dialect/MaxJ/IR/MaxJDialect.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ToolOutputFile.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

using namespace llvm;
using namespace mlir;
using namespace dfe;

static cl::opt<std::string>
    inputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"));

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"));

static cl::opt<bool>
    splitInputFile("split-input-file",
                   cl::desc("Split the input file into pieces and process each "
                            "chunk independently"),
                   cl::init(false));

static cl::opt<bool>
    verifyDiagnostics("verify-diagnostics",
                      cl::desc("Check that emitted diagnostics match "
                               "expected-* lines on the corresponding line"),
                      cl::init(false));

static cl::opt<bool>
    verifyPasses("verify-each",
                 cl::desc("Run the verifier after each transformation pass"),
                 cl::init(true));

static cl::opt<bool>
    showDialects("show-dialects",
                 cl::desc("Print the list of registered dialects"),
                 cl::init(false));

static cl::opt<bool> allowUnregisteredDialects(
    "allow-unregistered-dialect",
    cl::desc("Allow operation with no registered dialects"), cl::init(false));

int main(int argc, char *argv[]) {
  InitLLVM y(argc, argv);

  // register command line options
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerAsmPrinterCLOptions();

  // register the dialects
  registerDialect<StandardOpsDialect>();
  registerDialect<mlir::AffineDialect>();
  registerDialect<LLVM::LLVMDialect>();
  registerDialect<maxj::MaxJDialect>();
  initAffineToMaxJPasses();

  // parse passes
  PassPipelineCLParser passPipeline("", "Compiler passes to run");

  // parse options
  cl::ParseCommandLineOptions(argc, argv, "DFE optimizer\n");

  MLIRContext context;

  // Set up the input file.
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  // output file, by default to the stdout
  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    exit(1);
  }

  return failed(MlirOptMain(output->os(), std::move(file), passPipeline,
                            splitInputFile, verifyDiagnostics, verifyPasses,
                            allowUnregisteredDialects));

  return 0;
}