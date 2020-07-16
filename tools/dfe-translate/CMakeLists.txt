set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_tool(dfe-translate
  dfe-translate.cc
)

llvm_update_compile_flags(dfe-translate)

target_link_libraries(dfe-translate
  PRIVATE
  MLIRTranslation
  MLIRIR
  MLIRStandardOps
  MLIRSupport
 
  DFEEmitMaxJ
  # CIRCTFIRParser
  MLIRMaxJ
  MLIRDFETargetMaxJ
  )