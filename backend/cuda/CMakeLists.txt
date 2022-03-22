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
        vecmem::core vecmem::cuda
        ${CUDA_LIBRARIES})

install(TARGETS vecpar_cuda vecpar_core
        EXPORT vecpar_cuda
        LIBRARY DESTINATION ${CMAKE_INSTALL_CMAKEDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(EXPORT vecpar_cuda
        FILE vecpar.cmake
        NAMESPACE vecpar::
        DESTINATION ${CMAKE_INSTALL_CMAKEDIR})

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/vecpar
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})