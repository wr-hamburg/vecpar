include(CMakePackageConfigHelpers)
include(CPack)

set(CMAKE_INSTALL_CMAKEDIR
        "${CMAKE_INSTALL_LIBDIR}/cmake/vecpar-${PROJECT_VERSION}")

install(EXPORT vecpar-exports
        NAMESPACE "vecpar::"
        FILE "vecpar-config-targets.cmake"
        DESTINATION "${CMAKE_INSTALL_CMAKEDIR}" )

configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/vecpar-config.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/vecpar-config.cmake"
        INSTALL_DESTINATION "${CMAKE_INSTALL_CMAKEDIR}"
        PATH_VARS CMAKE_INSTALL_INCLUDEDIR CMAKE_INSTALL_LIBDIR
        CMAKE_INSTALL_CMAKEDIR
        NO_CHECK_REQUIRED_COMPONENTS_MACRO )

write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/vecpar-config-version.cmake"
        COMPATIBILITY "AnyNewerVersion" )

install( FILES
        "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/vecpar-config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/vecpar-config-version.cmake"
        DESTINATION "${CMAKE_INSTALL_CMAKEDIR}" )