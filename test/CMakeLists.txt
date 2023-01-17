### Test OMP code ###
if (VECPAR_BUILD_OMP_BACKEND)
    add_subdirectory(omp)
    add_subdirectory(ompt)
endif()

### Test CUDA code ###
if (VECPAR_BUILD_CUDA_BACKEND AND ${VECPAR_BUILD_WITH_CLANG})
    add_subdirectory(cuda)
endif()

### Test common interface that chooses from {CUDA,OMP} based on compiler ###
if (VECPAR_BUILD_OMP_BACKEND OR VECPAR_BUILD_CUDA_BACKEND)
    add_subdirectory(single_source)
endif()

if (VECPAR_BUILD_OMP_BACKEND AND VECPAR_BUILD_CUDA_BACKEND)
    add_subdirectory(hybrid)
endif()
