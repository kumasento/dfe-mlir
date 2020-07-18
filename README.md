# dfe-mlir

MLIR dialect for data-flow engine design

## Core Features

* A `maxj` dialect that can concisely represent most data-flow designs achievable by MaxJ (e.g., multi-kernel and LMem designs). We can perform optimization on it, and designs described by it can be finally translated to valid MaxJ code.

## Install

Please make sure your GCC and CMake satisfies the latest requirements from LLVM.

```shell
git clone --recursive https://github.com/kumasento/dfe-mlir

# build llvm
cd dfe-mlir/lib/llvm-project
mkdir build
cd build
cmake \
  -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;lld;mlir" \
  -DLLVM_TARGETS_TO_BUILD="X86" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_BUILD_LLVM_DYLIB=ON \
  -DLLVM_INCLUDE_TESTS=OFF -G "Ninja" ../llvm
ninja -j$(nproc)

# build dfe-mlir
cd ../../../
mkdir build
cd build
cmake -G "Ninja" ..
ninja
```

## Test

### The `maxj` Dialect