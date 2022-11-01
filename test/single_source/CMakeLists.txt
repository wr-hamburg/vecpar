### Managed memory ; compiled for CPU ###
if (${OpenMP_FOUND} AND ${CUDAToolkit_FOUND})
    add_executable(mm_tests_compiled_for_cpu "SingleSource_MM_Tests.cpp")

    set_target_properties(mm_tests_compiled_for_cpu PROPERTIES CXX_STANDARD 20)

    target_include_directories(mm_tests_compiled_for_cpu PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/common)

    target_compile_options(mm_tests_compiled_for_cpu PUBLIC
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
        $<$<COMPILE_LANGUAGE:CXX>:-DVECMEM_HAVE_PMR_MEMORY_RESOURCE>)

    target_link_libraries(mm_tests_compiled_for_cpu GTest::gtest_main
        vecpar_all
        vecmem::core vecmem::cuda)
endif()

### Managed memory ; compiled for GPU ###
if (VECPAR_BUILD_WITH_CLANG)
    add_executable(mm_tests_compiled_for_gpu "SingleSource_MM_Tests.cpp")

    target_include_directories(mm_tests_compiled_for_gpu PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/common)

    target_compile_options(mm_tests_compiled_for_gpu PUBLIC
        $<$<COMPILE_LANGUAGE:CXX>:-std=c++20 -Wall -Wextra -Wpedantic>
        $<$<COMPILE_LANGUAGE:CXX>:-x cuda --offload-arch=sm_86>
        $<$<COMPILE_LANGUAGE:CXX>:-DVECMEM_HAVE_PMR_MEMORY_RESOURCE>)

    target_link_libraries(mm_tests_compiled_for_gpu GTest::gtest_main
        vecpar_all)
endif()

### Host memory ; compiled for CPU ###
if (${OpenMP_FOUND})
    add_executable(hm_tests_compiled_for_cpu "SingleSource_HM_Tests.cpp")

    set_target_properties(hm_tests_compiled_for_cpu PROPERTIES CXX_STANDARD 20)

    target_include_directories(hm_tests_compiled_for_cpu PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/common)

    target_compile_options(hm_tests_compiled_for_cpu PUBLIC
            $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
            $<$<COMPILE_LANGUAGE:CXX>:-DVECMEM_HAVE_PMR_MEMORY_RESOURCE>)

    target_link_libraries(hm_tests_compiled_for_cpu GTest::gtest_main
            vecpar_all
            vecmem::core)
endif()

### Host memory ; compiled for GPU ###
if (VECPAR_BUILD_WITH_CLANG)
    add_executable(hm_tests_compiled_for_gpu "SingleSource_HM_Tests.cpp")

    target_include_directories(hm_tests_compiled_for_gpu PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/common)

    target_compile_options(hm_tests_compiled_for_gpu PUBLIC
            $<$<COMPILE_LANGUAGE:CXX>:-std=c++20 -Wall -Wextra -Wpedantic>
            $<$<COMPILE_LANGUAGE:CXX>:-x cuda --offload-arch=sm_86>
            $<$<COMPILE_LANGUAGE:CXX>:-DVECMEM_HAVE_PMR_MEMORY_RESOURCE>)

    target_link_libraries(hm_tests_compiled_for_gpu GTest::gtest_main
            vecpar_all)
endif()

# if compiled with OpenMP target support by gcc or clang compiler
#if (${OpenMP_FOUND} AND ${CUDAToolkit_FOUND})
 #   set(CMAKE_EXE_LINKER_FLAGS "-fopenmp")
 #   add_executable(hm_ompt_tests_compiled_for_gpu "SingleSource_HM_Tests.cpp")

  #  target_include_directories(hm_ompt_tests_compiled_for_gpu PUBLIC
   #         ${CMAKE_CURRENT_SOURCE_DIR}/common)

   # target_compile_options(hm_ompt_tests_compiled_for_gpu PUBLIC
    #        $<$<COMPILE_LANGUAGE:CXX>:-std=c++20 -fopenmp > #-foffload-options=nvptx-none=-latomic
     #       $<$<COMPILE_LANGUAGE:CXX>:-DVECMEM_HAVE_PMR_MEMORY_RESOURCE>)

   # target_link_libraries(hm_ompt_tests_compiled_for_gpu GTest::gtest_main
    #        vecpar_all)
#endif()