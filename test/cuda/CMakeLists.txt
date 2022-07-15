add_executable(tests_cuda "CudaTests.cpp")

target_include_directories(tests_cuda PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/common)

target_compile_options(tests_cuda PUBLIC
        $<$<COMPILE_LANGUAGE:CXX>:-std=c++20>
        $<$<COMPILE_LANGUAGE:CXX>:-x cuda --offload-arch=sm_86>
        $<$<COMPILE_LANGUAGE:CXX>:-DVECMEM_HAVE_PMR_MEMORY_RESOURCE>)

target_link_libraries(tests_cuda GTest::gtest_main
                      vecpar_cuda)

add_executable(overhead_cuda "benchmark/overhead_cuda.cpp")

target_include_directories(overhead_cuda PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/common)

target_compile_options(overhead_cuda PUBLIC
        $<$<COMPILE_LANGUAGE:CXX>:-std=c++20>
        $<$<COMPILE_LANGUAGE:CXX>:-x cuda --offload-arch=sm_86>
        $<$<COMPILE_LANGUAGE:CXX>:-DVECMEM_HAVE_PMR_MEMORY_RESOURCE>)

target_link_libraries(overhead_cuda vecpar_cuda)

