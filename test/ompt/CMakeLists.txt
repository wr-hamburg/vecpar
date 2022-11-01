### OMP target
add_executable(tests_ompt "OmptTests.cpp")

target_include_directories(tests_ompt PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/common)

set_target_properties(tests_ompt PROPERTIES CXX_STANDARD 20)

target_compile_options(tests_ompt PUBLIC
        $<$<COMPILE_LANGUAGE:CXX>:-DVECMEM_HAVE_PMR_MEMORY_RESOURCE -fopenmp >) # -foffload-options=nvptx-none=-latomic>)
# gcc -foffload=disable -foffload-options=nvptx-none=-latomic
# clang -fopenmp-targets=nvptx64 -fdebug-default-version=3 -v
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp")# -foffload-options=nvptx-none=-latomic") # -foffload=disable

target_link_libraries(tests_ompt GTest::gtest_main
        vecpar_ompt)

# -foffload-lto
