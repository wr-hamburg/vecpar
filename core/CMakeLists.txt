add_library(vecpar_core INTERFACE
        "include/vecpar/core/algorithms/detail/map.hpp"
        "include/vecpar/core/algorithms/detail/filter.hpp"
        "include/vecpar/core/algorithms/detail/reduce.hpp"
        "include/vecpar/core/algorithms/parallelizable_map_filter.hpp"
        "include/vecpar/core/algorithms/parallelizable_map.hpp"
        "include/vecpar/core/algorithms/parallelizable_reduce.hpp"
        "include/vecpar/core/algorithms/parallelizable_filter.hpp"
        "include/vecpar/core/definitions/common.hpp"
        "include/vecpar/core/definitions/config.hpp")

target_include_directories(vecpar_core INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_link_libraries(vecpar_core INTERFACE)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/vecpar
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})