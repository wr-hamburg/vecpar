add_executable(tests_omp "OmpTests.cpp")

target_include_directories(tests_omp PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/common)

target_compile_options(tests_omp PUBLIC
        $<$<COMPILE_LANGUAGE:CXX>:-std=c++20>
        $<$<COMPILE_LANGUAGE:CXX>:-DVECMEM_HAVE_PMR_MEMORY_RESOURCE>)

target_link_libraries(tests_omp GTest::gtest_main
                      vecpar_omp)


