add_mlir_dialect_library(MLIRDFE
  DFEDialect.cc

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Dialect/DFE

  DEPENDS
  MLIRDFEEnumsIncGen

  LINK_LIBS PUBLIC
  MLIREDSC
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRControlFlowInterfaces
  MLIRCallInterfaces
)