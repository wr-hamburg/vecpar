### OMP target
add_executable(tests_ompt "OmptTests.cpp")

target_include_directories(tests_ompt PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/common)

set_target_properties(tests_ompt PROPERTIES CXX_STANDARD 20)

# -foffload-lto == flag for clang link-time optimization; it should be added to both compile and link

target_compile_options(tests_ompt PUBLIC
        $<$<COMPILE_LANGUAGE:CXX>:-DVECMEM_HAVE_PMR_MEMORY_RESOURCE -fopenmp -fopenmp-targets=nvptx64>)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp -fopenmp-targets=nvptx64")

target_link_libraries(tests_ompt GTest::gtest_main
        vecpar_ompt OpenMP::OpenMP_CXX)


