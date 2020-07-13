set(LLVM_LINK_COMPONENTS
  Support
  )


add_llvm_tool(dfe-opt
  dfe-opt.cc
)
llvm_update_compile_flags(dfe-opt)

target_link_libraries(dfe-opt
  PRIVATE
  MLIRDFE

  MLIRParser
  MLIRSupport
  MLIRIR
  MLIROptLib
  MLIRStandardOps
  MLIRTransforms
  MLIRLLVMIR
  )