include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(GenerateExportHeader)

### set destination for header-only library ###
set(CMAKE_INSTALL_CMAKEDIR
        "${CMAKE_INSTALL_LIBDIR}/cmake/vecpar-${PROJECT_VERSION}")

function(install_library)
    install(FILES
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/vecpar-config.cmake
            DESTINATION ${CMAKE_INSTALL_CMAKEDIR})
endfunction()