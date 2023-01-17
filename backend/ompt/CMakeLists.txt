add_library(vecpar_ompt INTERFACE
        "include/vecpar/ompt/detail/config.hpp"
        "include/vecpar/ompt/ompt_parallelization.hpp")

target_include_directories(vecpar_ompt INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_link_libraries(vecpar_ompt INTERFACE
        vecpar_core vecmem::core OpenMP::OpenMP_CXX)

set_target_properties(vecpar_ompt PROPERTIES EXPORT_NAME ompt)
add_library(vecpar::ompt ALIAS vecpar_ompt)

install(TARGETS vecpar_ompt
        EXPORT vecpar-exports
        LIBRARY DESTINATION ${CMAKE_INSTALL_CMAKEDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/vecpar
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})