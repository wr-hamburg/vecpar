#include <gtest/gtest.h>

#include <vecmem/memory/cuda/managed_memory_resource.hpp>
#include <vecmem/containers/vector.hpp>

#include "../../common/infrastructure/TimeTest.hpp"
#include "../../common/infrastructure/sizes.hpp"

#include "../../common/algorithm/test_algorithm_1.hpp"
#include "../../common/algorithm/test_algorithm_2.hpp"
#include "../../common/algorithm/test_algorithm_3.hpp"
#include "../../common/algorithm/test_algorithm_4.hpp"
#include "../../common/algorithm/test_algorithm_5.hpp"

#include "vecpar/all/main.hpp"

namespace {

    class SingleSourceManagedMemoryTest : public TimeTest,
                                 public testing::WithParamInterface<int>{
    public:
        SingleSourceManagedMemoryTest() {
            vec = new vecmem::vector<int>(GetParam(), &mr);
            vec_d = new vecmem::vector<double>(GetParam(), &mr);
            for (size_t i = 0; i < vec->size(); i++) {
              vec->at(i) = i;
              vec_d->at(i) = i * 1.0;
              expectedReduceResult += i * 1.0;
              expectedFilterReduceResult += (i % 2 == 0) ? (i * 1.0) * 2 : 0;
            }
            printf("*******************************\n");
        }

    protected:
        vecmem::cuda::managed_memory_resource mr;
        vecmem::vector<int> *vec;
        vecmem::vector<double> *vec_d;
        double expectedReduceResult = 0;
        double expectedFilterReduceResult = 0;
    };

    TEST_P(SingleSourceManagedMemoryTest, Parallel_Map_Time) {
        std::chrono::time_point<std::chrono::steady_clock> start_time;
        std::chrono::time_point<std::chrono::steady_clock> end_time;

        test_algorithm_1 alg(mr);

        start_time = std::chrono::steady_clock::now();
        vecpar::parallel_map(alg, mr, *vec);
        end_time = std::chrono::steady_clock::now();

        std::chrono::duration<double> diff = end_time - start_time;
        printf("Parallel map time  = %f s\n", diff.count());
    }

    TEST_P(SingleSourceManagedMemoryTest, Parallel_Map_Correctness) {
        test_algorithm_1 alg(mr);
        vecmem::vector<double> result = vecpar::parallel_map(alg, mr, *vec);

        for (size_t i = 0; i < vec->size(); i++)
          EXPECT_EQ(vec->at(i) * 1.0, result.at(i));
    }

    TEST_P(SingleSourceManagedMemoryTest, Parallel_Reduce_Time) {
        std::chrono::time_point<std::chrono::steady_clock> start_time;
        std::chrono::time_point<std::chrono::steady_clock> end_time;

        test_algorithm_1 alg(mr);

        start_time = std::chrono::steady_clock::now();
        vecpar::parallel_reduce(alg, mr, *vec_d);
        end_time = std::chrono::steady_clock::now();

        std::chrono::duration<double> diff = end_time - start_time;
        printf("Parallel reduce <double> time  = %f s\n", diff.count());
    }

    TEST_P(SingleSourceManagedMemoryTest, Parallel_Filter_Time) {
        std::chrono::time_point<std::chrono::steady_clock> start_time;
        std::chrono::time_point<std::chrono::steady_clock> end_time;

        test_algorithm_3 alg(mr);

        start_time = std::chrono::steady_clock::now();
        vecpar::parallel_filter(alg, mr, *vec_d);
        end_time = std::chrono::steady_clock::now();

        std::chrono::duration<double> diff = end_time - start_time;
        printf("Parallel filter time  = %f s\n", diff.count());
    }

    TEST_P(SingleSourceManagedMemoryTest, Parallel_Filter_Correctness) {
        test_algorithm_3 alg(mr);

        vecmem::vector<double> result = vecpar::parallel_filter(alg, mr, *vec_d);

        int size = vec_d->size() % 2 == 0 ? int(vec_d->size()/2) : int(vec_d->size()/2) + 1;
        EXPECT_EQ(result.size(), size);

        // the order can be different
        std::sort(result.begin(), result.end());
        for (size_t i = 0; i < result.size(); i++) {
          EXPECT_EQ(vec_d->at(2 * i), result.at(i));
        }
    }

    TEST_P(SingleSourceManagedMemoryTest, Parallel_MapReduce_Separately) {
        test_algorithm_1 alg(mr);

        // parallel execution
        double result = vecpar::parallel_reduce(alg, mr,
                                                vecpar::parallel_map(alg, mr, *vec));

        EXPECT_EQ(result, expectedReduceResult);
    }

    TEST_P(SingleSourceManagedMemoryTest, Parallel_MapReduce_Grouped) {
        test_algorithm_1 alg(mr);

        // parallel execution
        double par_reduced = vecpar::parallel_algorithm(alg, mr, *vec);
        EXPECT_EQ(par_reduced, expectedReduceResult);
    }

    TEST_P(SingleSourceManagedMemoryTest, Parallel_Extra_Params_MapReduce_Separately) {
        test_algorithm_2 alg(mr);

        X x{1, 1.0};

        // parallel execution
        double result = vecpar::parallel_reduce(alg, mr,
                                                vecpar::parallel_map(alg, mr, *vec, x));

        EXPECT_EQ(result, expectedReduceResult);
    }

    TEST_P(SingleSourceManagedMemoryTest, Parallel_Extra_Params_MapReduce_Grouped) {
        test_algorithm_2 alg(mr);

        X x{1, 1.0};
        // parallel execution
        vecmem::vector<double> par_result(vec->size(), &mr);
        double par_reduced = vecpar::parallel_algorithm(alg, mr, *vec, x);
        EXPECT_EQ(par_reduced, expectedReduceResult);
    }


    TEST_P(SingleSourceManagedMemoryTest, Parallel_MapFilter_MapReduce_Chained) {
        test_algorithm_3 first_alg(mr);
        test_algorithm_4 second_alg;
        
        double second_result = vecpar::parallel_algorithm(second_alg, mr,
                                                          vecpar::parallel_algorithm(first_alg, mr, *vec));

        EXPECT_EQ(second_result, expectedFilterReduceResult);
    }

    TEST_P(SingleSourceManagedMemoryTest, Parallel_Map_Extra_Param) {
      test_algorithm_5 alg;

      X x{1, 1.0};
      // parallel execution + distructive change on the input!!!
      vecmem::vector<double> result = vecpar::parallel_map(alg, mr, *vec_d, x);
      EXPECT_EQ(result.size(), vec_d->size());
      for (size_t i = 0; i < result.size(); i++) {
        EXPECT_EQ(result.at(i), vec_d->at(i));
        EXPECT_EQ(result.at(i), (vec->at(i) + x.a) * x.b);
      }
    }


    INSTANTIATE_TEST_SUITE_P(Trivial_ManagedMemory, SingleSourceManagedMemoryTest, testing::ValuesIn(N));
}

