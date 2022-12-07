### OMP target for CPU
add_executable(tests_ompt_cpu "OmptTests.cpp")

target_include_directories(tests_ompt_cpu PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/common)

set_target_properties(tests_ompt_cpu PROPERTIES CXX_STANDARD 20)

target_compile_options(tests_ompt_cpu PUBLIC
        $<$<COMPILE_LANGUAGE:CXX>:-DVECMEM_HAVE_PMR_MEMORY_RESOURCE -fopenmp>)

target_link_libraries(tests_ompt_cpu GTest::gtest_main
        vecpar_ompt OpenMP::OpenMP_CXX)

target_link_options(tests_ompt_cpu PRIVATE -fopenmp -fopenmp-targets=x86_64 )

#----------------------------------------------------------#
### OMP target for GPU
add_executable(tests_ompt_gpu "OmptTests.cpp")

target_include_directories(tests_ompt_gpu PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/common)

set_target_properties(tests_ompt_gpu PROPERTIES CXX_STANDARD 20)

### Target options GPU
set(COMPILE_FLAGS "-DVECMEM_HAVE_PMR_MEMORY_RESOURCE -fopenmp")
set(LINK_FLAGS -fopenmp)

find_package(CUDAToolkit)
if (${CUDAToolkit_FOUND})
        set(COMPILE_FLAGS "${COMPILE_FLAGS} -fopenmp-targets=nvptx64")
        set(LINK_FLAGS ${LINK_FLAGS} -fopenmp-targets=nvptx64)
endif()

list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
find_package(hip QUIET)
if(hip_FOUND)
        set(COMPILE_FLAGS "${COMPILE_FLAGS} -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx1031")
        set(LINK_FLAGS "${LINK_FLAGS} -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx1031")
endif()

target_compile_options(tests_ompt_gpu PUBLIC
                $<$<COMPILE_LANGUAGE:CXX>:${COMPILE_FLAGS}>)
target_link_options(tests_ompt_gpu PRIVATE ${LINK_FLAGS})

#message(STATUS "FLAGS1: ^^^^^^^^^^^^^^^^ ${COMPILE_FLAGS}")
#message(STATUS "FLAGS2: ^^^^^^^^^^^^^^^^ ${LINK_FLAGS}")

target_link_libraries(tests_ompt_gpu GTest::gtest_main
        vecpar_ompt OpenMP::OpenMP_CXX)

#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -gline-tables-only -fopenmp-target-debug=3")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -foffload-lto")
#target_compile_options(tests_ompt_gpu PUBLIC        $<$<COMPILE_LANGUAGE:CXX>:-DVECMEM_HAVE_PMR_MEMORY_RESOURCE -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx1031 -v>) # -foffload-options=nvptx-none=-latomic>)
# gcc -foffload=disable -foffload-options=nvptx-none=-latomic
# clang -fopenmp-targets=nvptx64 -fdebug-default-version=3 -v
#target_link_options(tests_ompt_gpu PRIVATE ${CMAKE_EXE_LINKER_FLAGS} -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx1031)
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx1031")# -foffload-options=nvptx-none=-latomic") # -foffload=disable
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp -fopenmp-targets=nvptx64 -foffload-lto")
#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fopenmp -fopenmp-targets=nvptx64 ") #-Rpass=openmp-opt -gline-tables-only -fopenmp-target-debug=3

#target_link_options(tests_ompt_gpu PRIVATE ${CMAKE_EXE_LINKER_FLAGS} -fopenmp -fopenmp-targets=nvptx64 -foffload-lto)
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp -fopenmp-targets=nvptx64 ") #-foffload-lto -O3 ") #-Rpass=openmp-opt ")
#-Rpass=openmp-opt -Rpass-missed=openmp-opt -Rpass-analysis=openmp-opt

# FOR ALL EXECUTABLES
#enable clang link-time-optimization support
if (VECPAR_BUILD_WITH_CLANG)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -foffload-lto")
        target_link_options(tests_ompt_cpu PRIVATE -foffload-lto)
endif()