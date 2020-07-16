add_mlir_library(MLIRDFETargetMaxJ
  TranslateToMaxJ.cc

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/dfe/Target/MaxJ

  LINK_LIBS PUBLIC
  MLIRMaxJ
  )