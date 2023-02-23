#include <gtest/gtest.h>

#include <vecmem/containers/vector.hpp>
#include <vecmem/memory/host_memory_resource.hpp>

#include "../../common/infrastructure/TimeLogger.hpp"
#include "../../common/infrastructure/TimeTest.hpp"
#include "../../common/infrastructure/sizes.hpp"
#include "../../common/infrastructure/randoms.hpp"

#include "../../common/algorithm/benchmark/daxpy.hpp"
#include "../../common/algorithm/benchmark/saxpy.hpp"

#include "../../common/infrastructure/cleanup.hpp"
#include "vecpar/all/main.hpp"

namespace {
    class PerformanceTest_HostDevice : public TimeTest,
                                       public testing::WithParamInterface<int> {

    public:
        PerformanceTest_HostDevice() {
            vec = new vecmem::vector<int>(GetParam(), &mr);
            for (size_t i = 0; i < vec->size(); i++) {
                vec->at(i) = i;
                expectedReduceResult += i * 1.0;
            }
            printf("*******************************\n");
        }

        ~PerformanceTest_HostDevice() { cleanup::free(*vec); }

    protected:
        vecmem::host_memory_resource mr;
        vecmem::vector<int> *vec;
        double expectedReduceResult = 1.0;
        std::chrono::time_point<std::chrono::steady_clock> start_time;
        std::chrono::time_point<std::chrono::steady_clock> end_time;
    };

    void warmup() {
        int dev_num = omp_get_device_num();
        printf("Device num: %d\n", dev_num);
    }

    TEST_P(PerformanceTest_HostDevice, Saxpy) {
        saxpy alg;

        // arrays used for vecpar
        vecmem::vector<float> x_v(GetParam(), &mr);
        vecmem::vector<float> y_v(GetParam(), &mr);
        vecmem::vector<float> expected_result(GetParam(), &mr);

        float a = 2.0;

        // invoke the runtime
        warmup();

        // pseudo-random number generator
        std::mt19937 gen(rd());

        // init arrays
        for (int i = 0; i < GetParam(); i++) {
            x_v[i] = distro(gen);
            y_v[i] = distro(gen);
            expected_result[i] = y_v[i] + x_v[i] * a;
        }

        start_time = std::chrono::steady_clock::now();
        vecpar::ompt::parallel_algorithm(alg, mr, y_v, x_v, a);
        end_time = std::chrono::steady_clock::now();

        // check time
        std::chrono::duration<double> diff = end_time - start_time;
        printf("SAXPY vecpar OMPT time  = %f s\n", diff.count());

        // check results
        for (size_t i = 0; i < y_v.size(); i++) {
            EXPECT_EQ(y_v[i], expected_result[i]);
        }

#if defined(COMPILE_FOR_DEVICE)
        write_to_csv("gpu_saxpy_ompt_hd.csv", GetParam(), diff.count());
#else
        write_to_csv("cpu_saxpy_ompt_hd.csv", GetParam(), diff.count());
#endif
    }

    TEST_P(PerformanceTest_HostDevice, Daxpy) {
        daxpy alg;

        // arrays used for vecpar
        vecmem::vector<double> x_v(GetParam(), &mr);
        vecmem::vector<double> y_v(GetParam(), &mr);
        vecmem::vector<double> expected_result(GetParam(), &mr);

        double a = 2.0;

        // invoke the runtime
        warmup();

        // pseudo-random number generator
        std::mt19937 gen(rd());

        // init arrays
        for (int i = 0; i < GetParam(); i++) {
            x_v[i] = distro(gen);
            y_v[i] = distro(gen);
            expected_result[i] = y_v[i] + x_v[i] * a;
        }

        start_time = std::chrono::steady_clock::now();
        vecpar::ompt::parallel_algorithm(alg, mr, y_v, x_v, a);
        end_time = std::chrono::steady_clock::now();

        // check time
        std::chrono::duration<double> diff = end_time - start_time;
        printf("SAXPY vecpar OMPT time  = %f s\n", diff.count());

        // check results
        for (size_t i = 0; i < y_v.size(); i++) {
            EXPECT_EQ(y_v[i], expected_result[i]);
        }

#if defined(COMPILE_FOR_DEVICE)
        write_to_csv("gpu_daxpy_ompt_hd.csv", GetParam(), diff.count());
#else
        write_to_csv("cpu_daxpy_ompt_hd.csv", GetParam(), diff.count());
#endif
    }

    INSTANTIATE_TEST_SUITE_P(PerformanceTest_HostDevice, PerformanceTest_HostDevice,
                             testing::ValuesIn(N));
} // namespace
