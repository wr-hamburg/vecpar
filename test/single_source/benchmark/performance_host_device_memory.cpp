#include <gtest/gtest.h>

#include <vecmem/containers/vector.hpp>
#include <vecmem/memory/host_memory_resource.hpp>

#include "../../common/infrastructure/TimeLogger.hpp"
#include "../../common/infrastructure/TimeTest.hpp"
#include "../../common/infrastructure/sizes.hpp"
#include "../../common/infrastructure/randoms.hpp"

#include "../../common/algorithm/test_algorithm_3.hpp"
#include "../../common/algorithm/test_algorithm_4.hpp"
#include "../../common/algorithm/benchmark/saxpy.hpp"
#include "../../common/algorithm/benchmark/daxpy.hpp"

#include "vecpar/all/chain.hpp"
#include "vecpar/all/main.hpp"

namespace {
    class PerformanceTest_HostDevice : public TimeTest,
                                       public testing::WithParamInterface<int> {

    protected:
        vecmem::host_memory_resource mr;
        std::chrono::time_point<std::chrono::steady_clock> start_time;
        std::chrono::time_point<std::chrono::steady_clock> end_time;
    };

    TEST_P(PerformanceTest_HostDevice, Chain) {

        test_algorithm_3 first_alg(mr);
        test_algorithm_4 second_alg;

        // initialize arrays for default (benchmark) and chain calls
        vecmem::vector<int> vec_b(GetParam(), &mr);
        vecmem::vector<int> vec_c(GetParam(), &mr);

        //  init result
        double result_b = 0.0;
        double result_c = 0.0;

        // pseudo-random number generator
        std::mt19937 gen(rd());

        // init arrays
        for (int i = 0; i < GetParam(); i++) {
            // benchmark
            vec_b[i] = distro(gen);

            // chain
            vec_c[i] = vec_b[i];
        }

        start_time = std::chrono::steady_clock::now();
        result_b = vecpar::parallel_algorithm(second_alg, mr,
                                              vecpar::parallel_algorithm(first_alg, mr, vec_b));
        end_time = std::chrono::steady_clock::now();

        std::chrono::duration<double> diff1 = end_time - start_time;
        printf("Default = %f s\n", diff1.count());

        start_time = std::chrono::steady_clock::now();
        vecpar::chain<vecmem::host_memory_resource, double,
                vecmem::vector<int>>
                chain(mr);

        result_c = chain //.with_config(c)
                .with_algorithms(first_alg, second_alg)
                .execute(vec_c);
        end_time = std::chrono::steady_clock::now();

        std::chrono::duration<double> diff2 = end_time - start_time;
        printf("Chain  = %f s\n", diff2.count());

        // check results
        EXPECT_EQ(result_b, result_c);

#if defined(__CUDA__) && defined(__clang__)
        write_to_csv("gpu_chain_hd.csv", GetParam(), diff1.count(), diff2.count());
#else
        write_to_csv("cpu_chain_hd.csv", GetParam(), diff1.count(), diff2.count());
#endif
    }


#if defined(__CUDA__) && defined(__clang__)
#include "common.hpp"
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/utils/cuda/copy.hpp>

    vecmem::cuda::device_memory_resource d_mem;
    vecmem::cuda::copy copy;

    void warmup() {
        int deviceId;
        cudaGetDevice(&deviceId);
        printf("device id %d\n", deviceId);
    }

    template <class T>
    void benchmark(vecmem::vector<T> &x, vecmem::vector<T> &y, T a) {

      auto x_buffer =
          copy.to(vecmem::get_data(x), d_mem, vecmem::copy::type::host_to_device);
      auto x_view = vecmem::get_data(x_buffer);

      auto y_buffer =
          copy.to(vecmem::get_data(y), d_mem, vecmem::copy::type::host_to_device);
      auto y_view = vecmem::get_data(y_buffer);
      vecpar::config c = vecpar::cuda::getDefaultConfig(x.size());

      // call kernel
      kernel<T><<<c.m_gridSize, c.m_blockSize, c.m_memorySize>>>(x_view, y_view, a);

      CHECK_ERROR(cudaGetLastError());
      CHECK_ERROR(cudaDeviceSynchronize());

      copy(y_buffer, y, vecmem::copy::type::device_to_host);
    }
#else

#include "omp.h"

    void warmup() {
        int dev_num = omp_get_device_num();
        printf("Device num: %d\n", dev_num);
    }

    template<class T>
    void benchmark(vecmem::vector<T> &x, vecmem::vector<T> &y, T a) {
        int threadsNum = 1;
#pragma omp parallel for
        for (size_t i = 0; i < x.size(); i++) {
            y[i] = x[i] * a + y[i];
            DEBUG_ACTION(threadsNum = omp_get_num_threads();)
        }
        DEBUG_ACTION(printf("Using %d OpenMP threads \n", threadsNum);)
    }

#endif

    TEST_P(PerformanceTest_HostDevice, Saxpy) {
        saxpy alg;

        // arrays used for benchmark
        vecmem::vector<float> x_b(GetParam(), &mr);
        vecmem::vector<float> y_b(GetParam(), &mr);

        // arrays used for vecpar
        vecmem::vector<float> x_v(GetParam(), &mr);
        vecmem::vector<float> y_v(GetParam(), &mr);

        float a = 2.0;

        // invoke the runtime (openmp/cuda)
        warmup();

        // pseudo-random number generator
        std::mt19937 gen(rd());

        // init arrays
        for (int i = 0; i < GetParam(); i++) {
            // benchmark
            x_b[i] = distro(gen);
            y_b[i] = distro(gen);

            // vecpar
            x_v[i] = x_b[i];
            y_v[i] = y_b[i];
        }

        start_time = std::chrono::steady_clock::now();
        benchmark<float>(x_b, y_b, a);
        end_time = std::chrono::steady_clock::now();

        // check time
        std::chrono::duration<double> diff_benchmark = end_time - start_time;
        printf("SAXPY native time  = %f s\n", diff_benchmark.count());

        start_time = std::chrono::steady_clock::now();
        vecpar::parallel_algorithm(alg, mr, y_v, x_v, a);
        end_time = std::chrono::steady_clock::now();

        // check time
        std::chrono::duration<double> diff = end_time - start_time;
        printf("SAXPY vecpar time  = %f s\n", diff.count());

        // check results
        for (size_t i = 0; i < y_v.size(); i++) {
            EXPECT_EQ(y_b[i], y_v[i]);
        }

        // write results to csv file
#if defined(__CUDA__) && defined(__clang__)
        write_to_csv("gpu_saxpy_hd.csv", GetParam(), diff_benchmark.count(),
               diff.count());
#else
        write_to_csv("cpu_saxpy_hd.csv", GetParam(), diff_benchmark.count(),
                     diff.count());
#endif
    }

    TEST_P(PerformanceTest_HostDevice, Daxpy) {
        daxpy alg;

        // arrays used for benchmark
        vecmem::vector<double> x_b(GetParam(), &mr);
        vecmem::vector<double> y_b(GetParam(), &mr);

        // arrays used for vecpar
        vecmem::vector<double> x_v(GetParam(), &mr);
        vecmem::vector<double> y_v(GetParam(), &mr);

        double a = 2.0;

        // invoke the runtime (openmp/cuda)
        warmup();

        // pseudo-random number generator
        std::mt19937 gen(rd());

        // init arrays
        for (int i = 0; i < GetParam(); i++) {
            // benchmark
            x_b[i] = distro(gen);
            y_b[i] = distro(gen);

            // vecpar
            x_v[i] = x_b[i];
            y_v[i] = y_b[i];
        }

        start_time = std::chrono::steady_clock::now();
        benchmark<double>(x_b, y_b, a);
        end_time = std::chrono::steady_clock::now();

        // check time
        std::chrono::duration<double> diff_benchmark = end_time - start_time;
        printf("SAXPY native time  = %f s\n", diff_benchmark.count());

        start_time = std::chrono::steady_clock::now();
        vecpar::parallel_algorithm(alg, mr, y_v, x_v, a);
        end_time = std::chrono::steady_clock::now();

        // check time
        std::chrono::duration<double> diff = end_time - start_time;
        printf("SAXPY vecpar time  = %f s\n", diff.count());

        // check results
        for (size_t i = 0; i < y_v.size(); i++) {
            EXPECT_EQ(y_b[i], y_v[i]);
        }

        // write results to csv file
#if defined(__CUDA__) && defined(__clang__)
        write_to_csv("gpu_daxpy_hd.csv", GetParam(), diff_benchmark.count(),
               diff.count());
#else
        write_to_csv("cpu_daxpy_hd.csv", GetParam(), diff_benchmark.count(),
                     diff.count());
#endif
    }


    INSTANTIATE_TEST_SUITE_P(PerformanceTest_HostDevice, PerformanceTest_HostDevice,
                             testing::ValuesIn(N));
} // end namespace