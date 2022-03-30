### compiled for CPU ###
if (${OpenMP_FOUND})
    add_executable(tests_compiled_for_cpu "SingleSourceTests.cpp")

    target_include_directories(tests_compiled_for_cpu PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/common)

    target_compile_options(tests_compiled_for_cpu PUBLIC
        $<$<COMPILE_LANGUAGE:CXX>:-std=c++20>
        $<$<COMPILE_LANGUAGE:CXX>:-DVECMEM_HAVE_PMR_MEMORY_RESOURCE>)

    target_link_libraries(tests_compiled_for_cpu GTest::gtest_main
        vecpar_all
        vecmem::core vecmem::cuda)
endif()

### compiled for GPU ###
if (${CUDAToolkit_FOUND})
    add_executable(tests_compiled_for_gpu "SingleSourceTests.cpp")

    target_include_directories(tests_compiled_for_gpu PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/common)

    target_compile_options(tests_compiled_for_gpu PUBLIC
        $<$<COMPILE_LANGUAGE:CXX>:-std=c++20>
        $<$<COMPILE_LANGUAGE:CXX>:-x cuda --offload-arch=sm_86>
        $<$<COMPILE_LANGUAGE:CXX>:-DVECMEM_HAVE_PMR_MEMORY_RESOURCE>)

    target_link_libraries(tests_compiled_for_gpu GTest::gtest_main
        vecpar_all)
endif()
#-Wall -Wextra -Wpedantic
