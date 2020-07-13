add_mlir_dialect(DFE dfe)
add_mlir_doc(DFE -gen-op-doc dfe Dialect/)

set(LLVM_TARGET_DEFINITIONS DFE.td)
mlir_tablegen(DFEEnums.h.inc -gen-enum-decls)
mlir_tablegen(DFEEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRDFEEnumsIncGen)