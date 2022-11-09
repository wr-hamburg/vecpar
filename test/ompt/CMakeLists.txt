### OMP target for CPU
add_executable(tests_ompt_cpu "OmptTests.cpp")

target_include_directories(tests_ompt_cpu PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/common)

set_target_properties(tests_ompt_cpu PROPERTIES CXX_STANDARD 20)

target_compile_options(tests_ompt_cpu PUBLIC
        $<$<COMPILE_LANGUAGE:CXX>:-DVECMEM_HAVE_PMR_MEMORY_RESOURCE -fopenmp>)

# -foffload-lto == flag for clang link-time optimization; it should be added to both compile and link
# this is incompatible with debug -g flag
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -foffload-lto")

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp -fopenmp-targets=nvptx64 -foffload-lto")

target_link_libraries(tests_ompt_cpu GTest::gtest_main
        vecpar_ompt OpenMP::OpenMP_CXX)

### OMP target for GPU
add_executable(tests_ompt_gpu "OmptTests.cpp")

target_include_directories(tests_ompt_gpu PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/common)

set_target_properties(tests_ompt_gpu PROPERTIES CXX_STANDARD 20)

target_compile_options(tests_ompt_gpu PUBLIC
        $<$<COMPILE_LANGUAGE:CXX>:-DVECMEM_HAVE_PMR_MEMORY_RESOURCE -fopenmp -fopenmp-targets=nvptx64>)

# -foffload-lto == flag for clang link-time optimization; it should be added to both compile and link
# this is incompatible with debug -g flag
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -gline-tables-only -fopenmp-target-debug=3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -foffload-lto")

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp -fopenmp-targets=nvptx64 -foffload-lto")

target_link_libraries(tests_ompt_gpu GTest::gtest_main
        vecpar_ompt OpenMP::OpenMP_CXX)


