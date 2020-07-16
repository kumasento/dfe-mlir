cmake_minimum_required(VERSION 3.4.3) # to use find_package(LLVM)

if(POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()


# ---------------------------- Project 
project (dfe-mlir
         LANGUAGES C CXX
         VERSION 0.1.0
         DESCRIPTION "An experimental MLIR dialect for data-flow engine design")

# ---------------------------- C/C++ Setting
# Using C++ 14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# ---------------------------- Include
include (cmake/Macros.cmake)

# ---------------------------- Dependencies
include (cmake/LLVM.cmake) # find and include LLVM/MLIR

# ---------------------------- Main library
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

# for code generation purposes
add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(tools)

# ---------------------------- Testing
enable_testing()
# Use googletest 
add_subdirectory(lib/googletest)
include_directories(lib/googletest/googletest/include)