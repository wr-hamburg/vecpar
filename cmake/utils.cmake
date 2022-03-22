include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(GenerateExportHeader)

### set destination for header-only library ###
set(CMAKE_INSTALL_CMAKEDIR
        "${CMAKE_INSTALL_LIBDIR}/cmake/vecpar-${PROJECT_VERSION}")

function(install_library)
    add_library(${PROJECT_NAME} SHARED)
    target_sources(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/core/include
            $<IF:$<TARGET_EXISTS:vecpar_omp>,${CMAKE_CURRENT_SOURCE_DIR}/backend/omp/include,"">
            $<IF:$<TARGET_EXISTS:vecpar_cuda>,${CMAKE_CURRENT_SOURCE_DIR}/backend/cuda/include,"">
            $<IF:$<TARGET_EXISTS:vecpar_all>,${CMAKE_CURRENT_SOURCE_DIR}/backend/all/include,"">)

   set_target_properties(${PROJECT_NAME} PROPERTIES
           VERSION ${PROJECT_VERSION}
           SOVERSION ${PROJECT_VERSION}
           LINKER_LANGUAGE CXX)

    install(TARGETS ${PROJECT_NAME}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

    install(FILES
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/vecpar-config.cmake
            DESTINATION ${CMAKE_INSTALL_CMAKEDIR})
endfunction()