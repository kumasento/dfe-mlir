add_mlir_dialect_library(MLIRMaxJ
  MaxJDialect.cc
  MaxJOps.cc

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Dialect/MaxJ

  DEPENDS
  MLIRMaxJEnumsIncGen

  LINK_LIBS PUBLIC
  MLIREDSC
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRControlFlowInterfaces
  MLIRCallInterfaces
)