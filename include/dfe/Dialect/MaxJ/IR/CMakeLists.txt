add_mlir_dialect(MaxJ maxj)
add_mlir_doc(MaxJ -gen-op-doc maxj Dialect/)

set(LLVM_TARGET_DEFINITIONS MaxJ.td)
mlir_tablegen(MaxJEnums.h.inc -gen-enum-decls)
mlir_tablegen(MaxJEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRMaxJEnumsIncGen)