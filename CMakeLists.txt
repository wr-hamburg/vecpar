cmake_minimum_required(VERSION 3.18)

project(vecpar VERSION 0.0.1 DESCRIPTION
        "A header-only C++ library that provides parallelization support on top of vecmem library")

### include cmake helper functions ###
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/utils.cmake)

### build options ###
option(VECPAR_BUILD_OMP_BACKEND "Build OpenMP support for CPU parallelization" ON)
option(VECPAR_BUILD_CUDA_BACKEND "Build CUDA support for GPU offloading" ON)
option(VECPAR_BUILD_TESTS "Build automated tests" ON)

### dependencies for all the backends ###
find_package(LLVM 13.0.0 REQUIRED CONFIG)
find_package(Clang REQUIRED)
find_package(vecmem 0.10.0 REQUIRED)

### dependencies for OMP backend ###
if (VECPAR_BUILD_OMP_BACKEND)
    find_package(OpenMP REQUIRED)
    message(STATUS "Found OpenMP version ${OpenMP_CXX_VERSION}")
endif()

### dependencies for CUDA backend ###
if (VECPAR_BUILD_CUDA_BACKEND)
    find_package(CUDA REQUIRED 11.5)
    message(STATUS "Found CUDA ${CUDA_VERSION}")
endif()

### depencies for TEST ###
if (VECPAR_BUILD_TESTS)
    find_package(GTest REQUIRED)
    message(STATUS "Found GTest")
endif()

### include sources ###
add_subdirectory(core)
add_subdirectory(backend)

### include tests ###
if (VECPAR_BUILD_TESTS)
    add_subdirectory(test)
endif()

install_library()

