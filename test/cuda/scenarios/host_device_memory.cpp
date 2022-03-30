#include <gtest/gtest.h>

#include <cuda.h>

#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/containers/vector.hpp>

#include "../../common/infrastructure/TimeTest.hpp"
#include "../../common/infrastructure/sizes.hpp"

#include "../../common/algorithm/test_algorithm_1.hpp"
#include "../../common/algorithm/test_algorithm_2.hpp"
#include "../../common/algorithm/test_algorithm_3.hpp"
#include "../../common/algorithm/test_algorithm_4.hpp"
#include "../../common/algorithm/test_algorithm_5.hpp"

#include "vecpar/cuda/cuda_parallelization.hpp"

namespace {

    class GpuHostDeviceMemoryTest : public TimeTest,
                                    public testing::WithParamInterface<int>{
    public:
        GpuHostDeviceMemoryTest() {
            vec = new vecmem::vector<int>(GetParam(), &mr);
            vec_d = new vecmem::vector<double>(GetParam(), &mr);
            for (int i = 0; i < vec->size(); i++) {
                vec->at(i) = i;
                vec_d->at(i) = i * 1.0;
                expectedReduceResult += i * 1.0;
                expectedFilterReduceResult += (i % 2 == 0) ? (i * 1.0) * 2 : 0;
            }
            printf("*******************************\n");
        }

        ~GpuHostDeviceMemoryTest() {
            free(vec);
            free(vec_d);
        }

    protected:
        vecmem::host_memory_resource mr;
        vecmem::vector<int> *vec;
        vecmem::vector<double> *vec_d;
        double expectedReduceResult = 0;
        double expectedFilterReduceResult = 0;
    };

    TEST_P(GpuHostDeviceMemoryTest, Parallel_Inline_lambda) {
       X x{1,1.0};

       vecmem::cuda::device_memory_resource d_mem;
       vecmem::cuda::copy copy;

       // copy host to device
       auto vec_buffer = copy.to(vecmem::get_data(*vec), d_mem, vecmem::copy::type::host_to_device);

       vecpar::cuda::parallel_map(vec->size(), [=] __device__ (int idx,
                vecmem::data::vector_view<int>& vec_view) mutable {
                vecmem::device_vector<int> d_vec(vec_view);
                d_vec[idx] = d_vec[idx] * 4 + x.square_a();
            }, vecmem::get_data(vec_buffer));

       // copy device to host
       copy(vec_buffer, *vec, vecmem::copy::type::device_to_host);

       EXPECT_EQ(vec->at(0), 1.);
       EXPECT_EQ(vec->at(1), 5.);
       EXPECT_EQ(vec->at(2), 9.);
    }

    TEST_P(GpuHostDeviceMemoryTest, Parallel_Map_Time) {
        std::chrono::time_point<std::chrono::steady_clock> start_time;
        std::chrono::time_point<std::chrono::steady_clock> end_time;

        test_algorithm_1 alg(mr);

        start_time = std::chrono::steady_clock::now();
        vecmem::vector<double>* result = vecpar::cuda::parallel_map(alg, mr, *vec);
        end_time = std::chrono::steady_clock::now();

        std::chrono::duration<double> diff = end_time - start_time;
        printf("Parallel map time  = %f s\n", diff.count());
    }

    TEST_P(GpuHostDeviceMemoryTest, Parallel_Map_Correctness) {
        test_algorithm_1 alg(mr);
        vecmem::vector<double>* result = vecpar::cuda::parallel_map(alg, mr, *vec);

        for (int i = 0; i < vec->size(); i++)
            EXPECT_EQ(vec->at(i) * 1.0, result->at(i));
    }

    TEST_P(GpuHostDeviceMemoryTest, Parallel_Reduce_Time) {
        std::chrono::time_point<std::chrono::steady_clock> start_time;
        std::chrono::time_point<std::chrono::steady_clock> end_time;

        test_algorithm_1 alg(mr);

        start_time = std::chrono::steady_clock::now();
        double* result = vecpar::cuda::parallel_reduce(alg, mr, *vec_d);
        end_time = std::chrono::steady_clock::now();

        std::chrono::duration<double> diff = end_time - start_time;
        printf("Parallel reduce <double> time  = %f s\n", diff.count());
    }

    TEST_P(GpuHostDeviceMemoryTest, Parallel_Reduce_Correctness) {
        test_algorithm_1 alg(mr);
        double* result = vecpar::cuda::parallel_reduce(alg, mr, *vec_d);
        EXPECT_EQ(*result, expectedReduceResult);
    }

    TEST_P(GpuHostDeviceMemoryTest, Parallel_Filter_Time) {
        std::chrono::time_point<std::chrono::steady_clock> start_time;
        std::chrono::time_point<std::chrono::steady_clock> end_time;

        test_algorithm_3 alg(mr);

        start_time = std::chrono::steady_clock::now();
        vecmem::vector<double>* result = vecpar::cuda::parallel_filter(alg, mr, *vec_d);
        end_time = std::chrono::steady_clock::now();

        std::chrono::duration<double> diff = end_time - start_time;
        printf("Parallel filter time  = %f s\n", diff.count());
    }

    TEST_P(GpuHostDeviceMemoryTest, Parallel_Filter_Correctness) {
        test_algorithm_3 alg(mr);

        vecmem::vector<double>* result = vecpar::cuda::parallel_filter(alg, mr, *vec_d);

        int size = vec_d->size() % 2 == 0 ? int(vec_d->size()/2) : int(vec_d->size()/2) + 1;
        EXPECT_EQ(result->size(), size);

        // the order can be different
        std::sort(result->begin(), result->end());
        for (int i = 0; i < result->size(); i++) {
            EXPECT_EQ(vec_d->at(2 * i), result->at(i));
        }
    }

    TEST_P(GpuHostDeviceMemoryTest, Parallel_MapReduce_Grouped) {
        test_algorithm_1 alg(mr);

        // parallel execution
        double* par_reduced = vecpar::cuda::parallel_algorithm(alg, mr, *vec);
        EXPECT_EQ(*par_reduced, expectedReduceResult);
    }

    TEST_P(GpuHostDeviceMemoryTest, Parallel_Extra_Params_MapReduce_Grouped) {
        test_algorithm_2 alg(mr);

        X x{1, 1.0};
        // parallel execution
        double* par_reduced = vecpar::cuda::parallel_algorithm(alg, mr, *vec, x);
        EXPECT_EQ(*par_reduced, expectedReduceResult);
    }

    TEST_P(GpuHostDeviceMemoryTest, Parallel_MapFilter_MapReduce_Chained) {
        test_algorithm_3 first_alg(mr);
        test_algorithm_4 second_alg;

        vecmem::vector<double>* first_result = vecpar::cuda::parallel_algorithm(first_alg, mr, *vec);
        double* second_result = vecpar::cuda::parallel_algorithm(second_alg, mr, *first_result);

        EXPECT_EQ(*second_result, expectedFilterReduceResult);
    }

    TEST_P(GpuHostDeviceMemoryTest, Parallel_Map_Extra_Param) {
      test_algorithm_5 alg;

      X x{1, 1.0};
      // parallel execution + distructive change on the input!!!
      vecmem::vector<double> *result =
          vecpar::cuda::parallel_map(alg, mr, *vec_d, x);
      EXPECT_EQ(result->size(), vec_d->size());
      for (int i = 0; i < result->size(); i++) {
        EXPECT_EQ(result->at(i), vec_d->at(i));
        EXPECT_EQ(result->at(i), (vec->at(i) + x.a) * x.b);
        }
    }

    INSTANTIATE_TEST_SUITE_P(Trivial_HostDevice, GpuHostDeviceMemoryTest, testing::ValuesIn(N));
}
