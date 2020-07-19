# ------------------------------- Setup Variables
# LLVM is built in the current source tree
# Maybe we should add some error message later, i.e., build LLVM first

set(LLVM_DIR ${CMAKE_SOURCE_DIR}/lib/llvm-project/build/lib/cmake/llvm)
set(MLIR_DIR ${CMAKE_SOURCE_DIR}/lib/llvm-project/build/lib/cmake/mlir)

# ------------------------------- Find LLVM and MLIR
# This commit gives a better idea about how to integrate MLIR in cmake
# https://github.com/llvm/llvm-project/commit/7ca473a27bd589457d427eee9187d49a88fc9b01

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

set(CMAKE_MODULE_PATH
  ${LLVM_CMAKE_DIR}
  ${MLIR_CMAKE_DIR}
  )
include(AddLLVM)
include(TableGen)
include(AddMLIR)
include(HandleLLVMOptions)


llvm_map_components_to_libnames(LLVM_LIBS support core irreader)

set(MLIR_LIBS
  MLIRAnalysis
  MLIRIR
  MLIRParser
  MLIRSideEffectInterfaces
  MLIRTransforms)

# --------------------- LIT config
set(LIT_ARGS_DEFAULT "-sv")
set(LLVM_LIT_ARGS "${LIT_ARGS_DEFAULT}" CACHE STRING "Default options for lit")