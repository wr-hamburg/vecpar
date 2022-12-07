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


target_link_libraries(tests_ompt_cpu GTest::gtest_main
        vecpar_ompt OpenMP::OpenMP_CXX)

### OMP target for GPU
add_executable(tests_ompt_gpu "OmptTests.cpp")

target_include_directories(tests_ompt_gpu PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/common)

    set_target_properties(tests_ompt_gpu PROPERTIES CXX_STANDARD 20)

### Target options CPU
target_compile_options(tests_ompt_cpu PUBLIC
        $<$<COMPILE_LANGUAGE:CXX>:-DVECMEM_HAVE_PMR_MEMORY_RESOURCE -fopenmp>)
target_link_options(tests_ompt_cpu PRIVATE -fopenmp -fopenmp-targets=x86_64)


### Target options GPU
target_compile_options(tests_ompt_gpu PUBLIC
        $<$<COMPILE_LANGUAGE:CXX>:-DVECMEM_HAVE_PMR_MEMORY_RESOURCE -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx1031>)
target_link_options(tests_ompt_gpu PRIVATE -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx1031 )

#target_compile_options(tests_ompt_gpu PUBLIC
#        $<$<COMPILE_LANGUAGE:CXX>:-DVECMEM_HAVE_PMR_MEMORY_RESOURCE -fopenmp -fopenmp-targets=nvptx64>)
# -foffload-lto == flag for clang link-time optimization; it should be added to both compile and link
# this is incompatible with debug -g flag
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -gline-tables-only -fopenmp-target-debug=3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -foffload-lto")

#target_compile_options(tests_ompt_gpu PUBLIC        $<$<COMPILE_LANGUAGE:CXX>:-DVECMEM_HAVE_PMR_MEMORY_RESOURCE -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx1031 -v>) # -foffload-options=nvptx-none=-latomic>)

# gcc -foffload=disable -foffload-options=nvptx-none=-latomic
# clang -fopenmp-targets=nvptx64 -fdebug-default-version=3 -v
#target_link_options(tests_ompt_gpu PRIVATE ${CMAKE_EXE_LINKER_FLAGS} -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx1031)
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx1031")# -foffload-options=nvptx-none=-latomic") # -foffload=disable
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp -fopenmp-targets=nvptx64 -foffload-lto")

target_link_libraries(tests_ompt_gpu GTest::gtest_main
        vecpar_ompt OpenMP::OpenMP_CXX)


