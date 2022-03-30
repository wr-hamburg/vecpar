add_library(vecpar_all INTERFACE
        "include/vecpar/all/main.hpp")

target_include_directories(vecpar_all INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_link_libraries(vecpar_all INTERFACE
        $<TARGET_NAME_IF_EXISTS:vecpar_omp>
        $<TARGET_NAME_IF_EXISTS:vecpar_cuda>)

set_target_properties(vecpar_all PROPERTIES EXPORT_NAME all)
add_library(vecpar::all ALIAS vecpar_all)

install(TARGETS vecpar_all
        EXPORT vecpar-exports
        LIBRARY DESTINATION ${CMAKE_INSTALL_CMAKEDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/vecpar
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})