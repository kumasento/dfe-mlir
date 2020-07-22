add_mlir_conversion_library(DFEAffineToMaxJ
  AffineToMaxJ.cc

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/AffineToMaxJ

  DEPENDS
  DFEConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRAffineOps
  MLIRSCF
  MLIRPass
  MLIRStandardOps
  MLIRTransforms
  MLIRIR
  )