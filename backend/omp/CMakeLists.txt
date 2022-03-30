add_library(vecpar_omp INTERFACE
        "include/vecpar/omp/detail/internal.hpp"
        "include/vecpar/omp/omp_parallelization.hpp")

target_include_directories(vecpar_omp INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_link_libraries(vecpar_omp INTERFACE
        vecpar_core vecmem::core OpenMP::OpenMP_CXX)

set_target_properties(vecpar_omp PROPERTIES EXPORT_NAME omp)
add_library(vecpar::omp ALIAS vecpar_omp)

install(TARGETS vecpar_omp
        EXPORT vecpar-exports
        LIBRARY DESTINATION ${CMAKE_INSTALL_CMAKEDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/vecpar
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})