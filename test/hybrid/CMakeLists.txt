# the tests require vecmem::cuda for managed memory
if (CUDAToolkit_FOUND)
    add_executable(tests_hybrid "HybridTests.cpp")

    target_include_directories(tests_hybrid PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/common)

    set_target_properties(tests_hybrid PROPERTIES CXX_STANDARD 20)

    target_compile_options(tests_hybrid PUBLIC
        $<$<COMPILE_LANGUAGE:CXX>:-DVECMEM_HAVE_PMR_MEMORY_RESOURCE>)

    target_link_libraries(tests_hybrid GTest::gtest_main
                      vecpar_core vecpar_omp vecmem::cuda)
endif()
