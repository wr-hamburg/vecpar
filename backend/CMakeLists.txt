if (VECPAR_BUILD_OMP_BACKEND)
    add_subdirectory(omp)
endif()

if (VECPAR_BUILD_CUDA_BACKEND)
    add_subdirectory(cuda)
endif()

if (VECPAR_BUILD_OMP_BACKEND AND VECPAR_BUILD_CUDA_BACKEND)
    add_subdirectory(all)
endif()