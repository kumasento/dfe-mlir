file(GLOB globbed *.cc)

add_mlir_library(DFEEmitMaxJ
  ${globbed}
  
  ADDITIONAL_HEADER_DIRS

  LINK_LIBS PUBLIC
  MLIRMaxJ
  )