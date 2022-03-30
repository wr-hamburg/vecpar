add_library(vecpar_cuda INTERFACE
        "include/vecpar/cuda/detail/host_memory.hpp"
        "include/vecpar/cuda/detail/managed_memory.hpp"
        "include/vecpar/cuda/detail/internal.hpp"
        "include/vecpar/cuda/detail/kernels.hpp"
        "include/vecpar/cuda/detail/cuda_utils.hpp"
        "include/vecpar/cuda/cuda_parallelization.hpp")

target_include_directories(vecpar_cuda INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_link_libraries(vecpar_cuda INTERFACE
        vecpar_core
        vecmem::core vecmem::cuda CUDA::cudart)

set_target_properties(vecpar_cuda PROPERTIES EXPORT_NAME cuda)
add_library(vecpar::cuda ALIAS vecpar_cuda)

install(TARGETS vecpar_cuda
        EXPORT vecpar-exports
        LIBRARY DESTINATION ${CMAKE_INSTALL_CMAKEDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/vecpar
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})