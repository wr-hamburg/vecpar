add_executable(tests_omp "OmpTests.cpp")

target_include_directories(tests_omp PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/common)

set_target_properties(tests_omp PROPERTIES CXX_STANDARD 20)

target_compile_options(tests_omp PUBLIC
        $<$<COMPILE_LANGUAGE:CXX>:-DVECMEM_HAVE_PMR_MEMORY_RESOURCE>)

target_link_libraries(tests_omp GTest::gtest_main
                      vecpar_omp)

add_executable(overhead_omp "benchmark/overhead_omp.cpp")
target_include_directories(overhead_omp PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/common)

set_target_properties(overhead_omp PROPERTIES CXX_STANDARD 20)

target_compile_options(overhead_omp PUBLIC
        $<$<COMPILE_LANGUAGE:CXX>:-DVECMEM_HAVE_PMR_MEMORY_RESOURCE>)

target_link_libraries(overhead_omp vecpar_omp)

