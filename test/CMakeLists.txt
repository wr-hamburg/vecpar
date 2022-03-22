### Test OMP code ###
if (VECPAR_BUILD_OMP_BACKEND)
    add_subdirectory(omp)
endif()

### Test CUDA code ###
if (VECPAR_BUILD_CUDA_BACKEND)
    add_subdirectory(cuda)
endif()

### Test common interface that chooses from {CUDA,OMP} based on compiler ###
if (VECPAR_BUILD_OMP_BACKEND AND VECPAR_BUILD_CUDA_BACKEND)
    add_subdirectory(single_source)
    add_subdirectory(hybrid)
endif()

