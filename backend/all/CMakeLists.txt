add_library(vecpar_all INTERFACE
        "include/vecpar/all/main.hpp")

target_include_directories(vecpar_all INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_link_libraries(vecpar_all INTERFACE vecpar_omp vecpar_cuda OpenMP::OpenMP_CXX)

install(TARGETS vecpar_all vecpar_cuda vecpar_omp vecpar_core
        EXPORT vecpar_all
        LIBRARY DESTINATION ${CMAKE_INSTALL_CMAKEDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(EXPORT vecpar_all
        FILE vecpar.cmake
        NAMESPACE vecpar::
        DESTINATION ${CMAKE_INSTALL_CMAKEDIR})

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/vecpar
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})