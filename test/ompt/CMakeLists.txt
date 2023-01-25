### OMP target for CPU
add_executable(tests_ompt_cpu "OmptTests.cpp")
target_compile_definitions(tests_ompt_cpu PRIVATE COMPILE_FOR_HOST)

target_include_directories(tests_ompt_cpu PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/common)

set_target_properties(tests_ompt_cpu PROPERTIES CXX_STANDARD 20)

target_link_libraries(tests_ompt_cpu GTest::gtest_main
        vecpar_ompt OpenMP::OpenMP_CXX)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        target_compile_options(tests_ompt_cpu PUBLIC
                $<$<COMPILE_LANGUAGE:CXX>:-fopenmp -fopenmp-targets=x86_64>)
        target_link_options(tests_ompt_cpu PRIVATE -fopenmp -fopenmp-targets=x86_64 )
      #  target_link_options(tests_ompt_cpu PRIVATE -foffload-lto)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        target_compile_options(tests_ompt_cpu PUBLIC
                $<$<COMPILE_LANGUAGE:CXX>:-fopenmp -foffload=disable>)
        target_link_options(tests_ompt_cpu PRIVATE -fopenmp -foffload=disable)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AOMP")
        target_link_options(tests_ompt_cpu PRIVATE -fopenmp)
endif()

#----------------------------------------------------------#
### OMP target for GPU

#check if CUDA toolkit is available
find_package(CUDAToolkit QUIET)

#check if HIP toolkit is available
if(NOT DEFINED HIP_PATH)
        if(NOT DEFINED ENV{HIP_PATH})
                set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
        else()
                set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
        endif()
endif()
set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})
find_package(HIP QUIET)

# CUDA or HIP libraries are present, build the executable with offloading support
if (${CUDAToolkit_FOUND} OR ${HIP_FOUND})
        add_executable(tests_ompt_gpu "OmptTests.cpp")

        target_include_directories(tests_ompt_gpu PUBLIC
                ${CMAKE_CURRENT_SOURCE_DIR}/common)

        set_target_properties(tests_ompt_gpu PROPERTIES CXX_STANDARD 20)

        ### Target options GPU
        set(COMPILE_FLAGS "-fopenmp")
        set(LINK_FLAGS -fopenmp)
        target_compile_definitions(tests_ompt_gpu PRIVATE COMPILE_FOR_DEVICE)

        if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
                # NVIDIA flags
                set(COMPILE_FLAGS ${COMPILE_FLAGS} -fopenmp-targets=nvptx64 )
                set(LINK_FLAGS ${LINK_FLAGS} -fopenmp-targets=nvptx64 )
                # AMD flags
                #set(COMPILE_FLAGS ${COMPILE_FLAGS} -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx1031)
                #set(LINK_FLAGS ${LINK_FLAGS} -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx1031)
        elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
                set(COMPILE_FLAGS ${COMPILE_FLAGS} -foffload=nvptx-none)
                set(LINK_FLAGS ${LINK_FLAGS} -foffload=nvptx-none)
        elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AOMP")
                set(COMPILE_FLAGS ${COMPILE_FLAGS} -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx1031)
                set(LINK_FLAGS ${LINK_FLAGS} -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx1031)
        endif()

        target_compile_options(tests_ompt_gpu PUBLIC
                $<$<COMPILE_LANGUAGE:CXX>:${COMPILE_FLAGS}>)
        target_link_options(tests_ompt_gpu PRIVATE ${LINK_FLAGS})

        target_link_libraries(tests_ompt_gpu GTest::gtest_main
                vecpar_ompt OpenMP::OpenMP_CXX)

        if (VECPAR_BUILD_WITH_CLANG)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -foffload-lto")
                target_link_options(tests_ompt_gpu PRIVATE -foffload-lto)
        endif()
endif()