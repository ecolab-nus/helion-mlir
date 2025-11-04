cmake_minimum_required(VERSION 3.20)

project(json2mlir LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_RUNTIME_OUTPUT_DIRECTORY)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()

find_package(MLIR REQUIRED CONFIG HINTS $ENV{MLIR_DIR})

message(STATUS "Configuring json2mlir with MLIR at ${MLIR_DIR}")
message(STATUS "MLIR include dirs: ${MLIR_INCLUDE_DIRS}")
message(STATUS "LLVM include dirs: ${LLVM_INCLUDE_DIRS}")

set(LLVM_NO_DEAD_STRIP 1)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

add_subdirectory(lib)
