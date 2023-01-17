#include <gtest/gtest.h>

#include <vecmem/containers/vector.hpp>
#include <vecmem/memory/cuda/managed_memory_resource.hpp>

#include "../common/infrastructure/TimeTest.hpp"
#include "../common/infrastructure/sizes.hpp"

#include "../common/algorithm/test_algorithm_1.hpp"
#include "../common/algorithm/test_algorithm_2.hpp"
#include "../common/algorithm/test_algorithm_3.hpp"
#include "../common/algorithm/test_algorithm_4.hpp"

#include "../common/infrastructure/cleanup.hpp"
#include "vecpar/omp/omp_parallelization.hpp"

namespace {

class CpuManagedMemoryTest : public TimeTest,
                             public testing::WithParamInterface<int> {
public:
  CpuManagedMemoryTest() {
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

  ~CpuManagedMemoryTest() {
    cleanup::free(*vec);
    cleanup::free(*vec_d);
  }

protected:
  vecmem::cuda::managed_memory_resource mr;
  vecmem::vector<int> *vec;
  vecmem::vector<double> *vec_d;
  double expectedReduceResult = 0;
  double expectedFilterReduceResult = 0;
};

TEST_P(CpuManagedMemoryTest, Parallel_MapOnly) {
  test_algorithm_1 alg;

  vecmem::vector<double> par_result(vec->size(), &mr);

  vecpar::omp::parallel_map(vec->size(), [&] (int idx) mutable {
    alg.map(par_result[idx], vec->at(idx));
  });

  EXPECT_EQ(par_result[0], vec->at(0));
  EXPECT_EQ(par_result[int(GetParam() / 2)], vec->at(int(GetParam() / 2)));
  EXPECT_EQ(par_result[GetParam() - 1], vec->at(GetParam() - 1));
}

TEST_P(CpuManagedMemoryTest, Parallel_Inline_lambda) {
  test_algorithm_1 alg;

  vecpar::omp::parallel_map(
      vec->size(), [&] (int idx) mutable { vec->at(idx) *= 4.0; });

  EXPECT_EQ(vec->at(0), 0);
  EXPECT_EQ(vec->at(1), 4.);
  EXPECT_EQ(vec->at(2), 8.);
}

TEST_P(CpuManagedMemoryTest, Parallel_Map_Time) {
  std::chrono::time_point<std::chrono::steady_clock> start_time;
  std::chrono::time_point<std::chrono::steady_clock> end_time;

  test_algorithm_1 alg;

  start_time = std::chrono::steady_clock::now();
  vecmem::vector<double> result = vecpar::omp::parallel_map(alg, mr, *vec);
  end_time = std::chrono::steady_clock::now();

  std::chrono::duration<double> diff = end_time - start_time;
  printf("Parallel map time  = %f s\n", diff.count());
}

TEST_P(CpuManagedMemoryTest, Parallel_Map_Correctness) {
  test_algorithm_1 alg;
  vecmem::vector<double> result = vecpar::omp::parallel_map(alg, mr, *vec);

  for (int i = 0; i < vec->size(); i++)
    EXPECT_EQ(vec->at(i) * 1.0, result.at(i));
}

TEST_P(CpuManagedMemoryTest, Parallel_Reduce_Time) {
  std::chrono::time_point<std::chrono::steady_clock> start_time;
  std::chrono::time_point<std::chrono::steady_clock> end_time;

  test_algorithm_1 alg;

  start_time = std::chrono::steady_clock::now();
  vecpar::omp::parallel_reduce(alg, mr, *vec_d);
  end_time = std::chrono::steady_clock::now();

  std::chrono::duration<double> diff = end_time - start_time;
  printf("Parallel reduce <double> time  = %f s\n", diff.count());
}

TEST_P(CpuManagedMemoryTest, Parallel_Reduce_Correctness) {
  test_algorithm_1 alg;
  double result = vecpar::omp::parallel_reduce(alg, mr, *vec_d);
  EXPECT_EQ(result, expectedReduceResult);
}

TEST_P(CpuManagedMemoryTest, Parallel_Filter_Time) {
  std::chrono::time_point<std::chrono::steady_clock> start_time;
  std::chrono::time_point<std::chrono::steady_clock> end_time;

  test_algorithm_3 alg(mr);

  start_time = std::chrono::steady_clock::now();
  vecmem::vector<double> result = vecpar::omp::parallel_filter(alg, mr, *vec_d);
  end_time = std::chrono::steady_clock::now();

  std::chrono::duration<double> diff = end_time - start_time;
  printf("Parallel filter time  = %f s\n", diff.count());
}

TEST_P(CpuManagedMemoryTest, Parallel_Filter_Correctness) {
  test_algorithm_3 alg(mr);

  vecmem::vector<double> result = vecpar::omp::parallel_filter(alg, mr, *vec_d);

  int size = vec_d->size() % 2 == 0 ? int(vec_d->size() / 2)
                                    : int(vec_d->size() / 2) + 1;
  EXPECT_EQ(result.size(), size);

  // the order can be different
  std::sort(result.begin(), result.end());
  for (int i = 0; i < result.size(); i++) {
    EXPECT_EQ(vec_d->at(2 * i), result.at(i));
  }
}

TEST_P(CpuManagedMemoryTest, Serial_MapReduce) {
  test_algorithm_1 alg;

  // serial execution
  double *result = alg(*vec);
  EXPECT_EQ(*result, expectedReduceResult);
}

TEST_P(CpuManagedMemoryTest, Parallel_MapReduce_Separately) {
  test_algorithm_1 alg;

  // parallel execution
  vecmem::vector<double> par_result(vec->size(), &mr);

  vecpar::omp::parallel_map(vec->size(), [&] (int idx) mutable {
    alg.map(par_result[idx], vec->at(idx));
  });

  double *result = new double();
  vecpar::omp::parallel_reduce(
      vec->size(), result,
      [&] (double *r, double tmp) mutable { alg.reduce(r, tmp); },
      par_result);

  EXPECT_EQ(*result, expectedReduceResult);
}

TEST_P(CpuManagedMemoryTest, Parallel_MapReduce_Grouped) {
  test_algorithm_1 alg;

  // parallel execution
  vecmem::vector<double> par_result(vec->size(), &mr);
  double par_reduced = vecpar::omp::parallel_algorithm(alg, mr, *vec);
  EXPECT_EQ(par_reduced, expectedReduceResult);
}

TEST_P(CpuManagedMemoryTest, Serial_MapReduce_Extra_Params) {
  test_algorithm_2 alg;
  X x{1, 1.0};
  // serial execution
  double *result = alg(*vec, x);
  EXPECT_EQ(*result, expectedReduceResult);
}

TEST_P(CpuManagedMemoryTest, Parallel_Extra_Params_MapReduce_Separately) {
  test_algorithm_2 alg;

  X x{1, 1.0};

  // parallel execution
  vecmem::vector<double> par_result(vec->size(), &mr);

  vecpar::omp::parallel_map(vec->size(), [&] (int idx) mutable {
    alg.map(par_result[idx], vec->at(idx), x);
  });

  double *result = new double();
  vecpar::omp::parallel_reduce(
      vec->size(), result,
      [&] (double *r, double tmp) mutable { alg.reduce(r, tmp); },
      par_result);

  EXPECT_EQ(*result, expectedReduceResult);
}

TEST_P(CpuManagedMemoryTest, Parallel_Extra_Params_MapReduce_Grouped) {
  test_algorithm_2 alg;

  X x{1, 1.0};
  // parallel execution
  vecmem::vector<double> par_result(vec->size(), &mr);
  double par_reduced = vecpar::omp::parallel_algorithm(alg, mr, *vec, x);
  EXPECT_EQ(par_reduced, expectedReduceResult);
}

TEST_P(CpuManagedMemoryTest, Serial_MapFilter_MapReduce_Chained) {
  test_algorithm_3 first_alg(mr);
  test_algorithm_4 second_alg;

  vecmem::vector<double> first_result = first_alg(*vec);
  double *second_result = second_alg(first_result);

  EXPECT_EQ(*second_result, expectedFilterReduceResult);
}

TEST_P(CpuManagedMemoryTest, Parallel_MapFilter_MapReduce_Chained) {
  test_algorithm_3 first_alg(mr);
  test_algorithm_4 second_alg;

  vecmem::vector<double> first_result =
      vecpar::omp::parallel_algorithm(first_alg, mr, *vec);
  double second_result =
      vecpar::omp::parallel_algorithm(second_alg, mr, first_result);

  EXPECT_EQ(second_result, expectedFilterReduceResult);
}

INSTANTIATE_TEST_SUITE_P(OMP_ManagedMemory, CpuManagedMemoryTest,
                         testing::ValuesIn(N));
} // namespace
