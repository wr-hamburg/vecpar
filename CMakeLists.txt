cmake_minimum_required(VERSION 3.19)

project(vecpar VERSION 0.0.1 DESCRIPTION
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
if (VECPAR_BUILD_OMP_BACKEND)
    find_package(OpenMP REQUIRED)
   # message(STATUS "Found OpenMP version ${OpenMP_CXX_VERSION}")
endif()

### dependencies for CUDA backend ###
if (VECPAR_BUILD_CUDA_BACKEND)
    find_package(CUDAToolkit REQUIRED 11.5)
   # message(STATUS "Found CUDATookit ${CUDAToolkit_VERSION}")
endif()

### depencies for TEST ###
if (VECPAR_BUILD_TESTS)
    find_package(GTest REQUIRED)
#    message(STATUS "Found GTest")
endif()

set(VECPAR_BUILD_WITH_CLANG False)
if (CUDAToolkit_FOUND)
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        set(VECPAR_BUILD_WITH_CLANG True)
    endif()
endif()

### include sources ###
add_subdirectory(core)
add_subdirectory(backend)

### include tests ###
if (VECPAR_BUILD_TESTS)
    add_subdirectory(test)
endif()

### export targets ###
include(vecpar-pack)