cmake_minimum_required(VERSION 3.19)

project(vecpar VERSION 0.0.2 DESCRIPTION
        "A header-only C++ library that provides parallelization support on top of vecmem library")

include(GNUInstallDirs)

list( APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" )

### build options ###
option(VECPAR_BUILD_OMP_BACKEND "Build OpenMP support for CPU parallelization" ON)
option(VECPAR_BUILD_CUDA_BACKEND "Build CUDA support for GPU offloading" ON)
option(VECPAR_BUILD_TESTS "Build automated tests" ON)

### dependencies for all the backends ###
#find_package(LLVM REQUIRED)
#find_package(Clang REQUIRED)

# Set up VecMem.
option(VECPAR_SETUP_VECMEM
        "Set up the VecMem target(s) explicitly" TRUE )
option(VECPAR_USE_SYSTEM_VECMEM
        "Pick up an existing installation of VecMem from the build environment"
        TRUE )

if(VECPAR_SETUP_VECMEM )
    if(VECPAR_USE_SYSTEM_VECMEM )
        find_package( vecmem 0.10.0 REQUIRED )
    else()
        # BUG: this doesn't work due to the compiler flags "-x cuda" required by certain tests
       # add_subdirectory( extern/vecmem )
    endif()
endif()

### dependencies for OMP backend ###
find_package(OpenMP)
set( VECPAR_BUILD_OMP_BACKEND ${OpenMP_FOUND} CACHE BOOL
        "Enable/disable vecpar OpenMP backend")

### dependencies for CUDA backend ###
find_package(CUDAToolkit 11.5)
set( VECPAR_BUILD_CUDA_BACKEND ${CUDA_FOUND} CACHE BOOL
        "Enable/disable vecpar CUDA backend")

### dependencies for TEST ###
find_package(GTest)
set( VECPAR_BUILD_TESTS ${GTest_FOUND} CACHE BOOL
        "Enable/disable vecpar tests")

set(VECPAR_BUILD_WITH_CLANG False)
if (CUDAToolkit_FOUND)
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        set(VECPAR_BUILD_WITH_CLANG True)
    endif()
endif()

### include sources ###
add_subdirectory(core)
add_subdirectory(backend)
add_subdirectory(extern/composition)

### include tests ###
if (VECPAR_BUILD_TESTS)
    add_subdirectory(test)
endif()

### export targets ###
include(vecpar-pack)