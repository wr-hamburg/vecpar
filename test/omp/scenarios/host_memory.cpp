#include <gtest/gtest.h>

#include <vecmem/containers/vector.hpp>
#include <vecmem/memory/host_memory_resource.hpp>

#include "../../common/infrastructure/TimeTest.hpp"
#include "../../common/infrastructure/sizes.hpp"

#include "../../common/algorithm/test_algorithm_1.hpp"
#include "../../common/algorithm/test_algorithm_2.hpp"
#include "../../common/algorithm/test_algorithm_3.hpp"
#include "../../common/algorithm/test_algorithm_4.hpp"
#include "../../common/algorithm/test_algorithm_5.hpp"

#include "../native_algorithms/test_algorithm_2_omp.hpp"
#include "../native_algorithms/test_algorithm_2_omp_optimized.hpp"
#include "../native_algorithms/test_algorithm_2_seq.hpp"

#include "vecpar/omp/omp_parallelization.hpp"

namespace {

class CpuHostMemoryTest : public TimeTest,
                          public testing::WithParamInterface<int> {
public:
  CpuHostMemoryTest() {
    vec = new vecmem::vector<int>(GetParam(), &mr);
    vec_d = new vecmem::vector<double>(GetParam(), &mr);

    for (int i = 0; i < vec->size(); i++) {
      vec->at(i) = i;
      vec_d->at(i) = i * 1.0;
      expectedReduceResult += vec_d->at(i);
      expectedFilterReduceResult += (i % 2 == 0) ? (i * 1.0) * 2 : 0;
    }
    printf("*******************************\n");
  }

protected:
  vecmem::host_memory_resource mr;
  vecmem::vector<int> *vec;
  vecmem::vector<double> *vec_d;
  double expectedReduceResult = 0;
  double expectedFilterReduceResult = 0;
};

TEST_P(CpuHostMemoryTest, Parallel_MapOnly) {
  test_algorithm_1 alg(mr);

  vecmem::vector<double> par_result(vec->size(), &mr);

  vecpar::omp::parallel_map(vec->size(), [&] TARGET(int idx) mutable {
    alg.map(par_result[idx], vec->at(idx));
  });

  EXPECT_EQ(par_result[0], vec->at(0));
  EXPECT_EQ(par_result[int(GetParam() / 2)], vec->at(int(GetParam() / 2)));
  EXPECT_EQ(par_result[GetParam() - 1], vec->at(GetParam() - 1));
}

TEST_P(CpuHostMemoryTest, Parallel_ReduceOnly) {
  test_algorithm_1 alg(mr);

  double *result = new double();
  vecpar::omp::parallel_reduce(
      vec->size(), result,
      [&] TARGET(double *r, double tmp) mutable { alg.reduce(r, tmp); },
      *vec_d);

  EXPECT_EQ(*result, expectedReduceResult);
}

TEST_P(CpuHostMemoryTest, Parallel_Inline_lambda) {
  test_algorithm_1 alg(mr);

  vecpar::omp::parallel_map(
      vec->size(), [&] TARGET(int idx) mutable { vec->at(idx) *= 4.0; });

  EXPECT_EQ(vec->at(0), 0);
  EXPECT_EQ(vec->at(1), 4.);
  EXPECT_EQ(vec->at(2), 8.);
}

TEST_P(CpuHostMemoryTest, Parallel_Map_Time) {
  std::chrono::time_point<std::chrono::steady_clock> start_time;
  std::chrono::time_point<std::chrono::steady_clock> end_time;

  test_algorithm_1 alg(mr);

  start_time = std::chrono::steady_clock::now();
  vecmem::vector<double> result = vecpar::omp::parallel_map(alg, mr, *vec);
  end_time = std::chrono::steady_clock::now();

  std::chrono::duration<double> diff = end_time - start_time;
  printf("Parallel map time  = %f s\n", diff.count());
}

TEST_P(CpuHostMemoryTest, Parallel_Map_Correctness) {
  test_algorithm_1 alg(mr);
  vecmem::vector<double> result = vecpar::omp::parallel_map(alg, mr, *vec);

  for (int i = 0; i < vec->size(); i++)
    EXPECT_EQ(vec->at(i) * 1.0, result.at(i));
}

TEST_P(CpuHostMemoryTest, Parallel_Reduce_Time) {
  std::chrono::time_point<std::chrono::steady_clock> start_time;
  std::chrono::time_point<std::chrono::steady_clock> end_time;

  test_algorithm_1 alg(mr);

  start_time = std::chrono::steady_clock::now();
  double result = vecpar::omp::parallel_reduce(alg, mr, *vec_d);
  end_time = std::chrono::steady_clock::now();

  std::chrono::duration<double> diff = end_time - start_time;
  printf("Parallel reduce <double> time  = %f s\n", diff.count());
}

TEST_P(CpuHostMemoryTest, Parallel_Reduce_Correctness) {
  test_algorithm_1 alg(mr);
  double result = vecpar::omp::parallel_reduce(alg, mr, *vec_d);
  EXPECT_EQ(result, expectedReduceResult);
}

TEST_P(CpuHostMemoryTest, Parallel_Filter_Time) {
  std::chrono::time_point<std::chrono::steady_clock> start_time;
  std::chrono::time_point<std::chrono::steady_clock> end_time;

  test_algorithm_3 alg(mr);

  start_time = std::chrono::steady_clock::now();
  vecmem::vector<double> result = vecpar::omp::parallel_filter(alg, mr, *vec_d);
  end_time = std::chrono::steady_clock::now();

  std::chrono::duration<double> diff = end_time - start_time;
  printf("Parallel filter time  = %f s\n", diff.count());
}

TEST_P(CpuHostMemoryTest, Parallel_Filter_Correctness) {
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

TEST_P(CpuHostMemoryTest, Serial_MapReduce) {
  test_algorithm_1 alg(mr);

  // serial execution
  double *result = alg(*vec);
  EXPECT_EQ(*result, expectedReduceResult);
}

TEST_P(CpuHostMemoryTest, Parallel_MapReduce_Separately) {
  test_algorithm_1 alg(mr);

  // parallel execution
  vecmem::vector<double> par_result(vec->size(), &mr);

  vecpar::omp::parallel_map(vec->size(), [&] TARGET(int idx) mutable {
    alg.map(par_result[idx], vec->at(idx));
  });

  double *result = new double();
  vecpar::omp::parallel_reduce(
      vec->size(), result,
      [&] TARGET(double *r, double tmp) mutable { alg.reduce(r, tmp); },
      par_result);

  EXPECT_EQ(*result, expectedReduceResult);
}

TEST_P(CpuHostMemoryTest, Parallel_MapReduce_Grouped) {
  test_algorithm_1 alg(mr);

  // parallel execution
  double par_reduced = vecpar::omp::parallel_algorithm(alg, mr, *vec);
  EXPECT_EQ(par_reduced, expectedReduceResult);
}

TEST_P(CpuHostMemoryTest, Serial_Extra_Params_MapReduce) {
  test_algorithm_2 alg(mr);
  X x{1, 1.0};

  // serial execution
  double *result = alg(*vec, x);
  EXPECT_EQ(*result, expectedReduceResult);
}

TEST_P(CpuHostMemoryTest, Parallel_Extra_Params_MapReduce_Separately) {
  test_algorithm_2 alg(mr);

  X x{1, 1.0};

  // parallel execution
  vecmem::vector<double> par_result(vec->size(), &mr);

  vecpar::omp::parallel_map(vec->size(), [&] TARGET(int idx) mutable {
    alg.map(par_result[idx], vec->at(idx), x);
  });

  double *result = new double();
  vecpar::omp::parallel_reduce(
      vec->size(), result,
      [&] TARGET(double *r, double tmp) mutable { alg.reduce(r, tmp); },
      par_result);

  EXPECT_EQ(*result, expectedReduceResult);
}

TEST_P(CpuHostMemoryTest, Parallel_Extra_Params_MapReduce_Grouped) {
  test_algorithm_2 alg(mr);

  X x{1, 1.0};
  // parallel execution
  double par_reduced = vecpar::omp::parallel_algorithm(alg, mr, *vec, x);
  EXPECT_EQ(par_reduced, expectedReduceResult);
}

TEST_P(CpuHostMemoryTest, Parallel_Extra_Params_MapReduce_op_OMP) {
  test_algorithm_2_omp alg_test(mr);
  X x{1, 1.0};

  // parallel execution
  double par_reduced = alg_test(*vec, x);
  EXPECT_EQ(par_reduced, expectedReduceResult);
}

TEST_P(CpuHostMemoryTest, Parallel_Extra_Params_MapReduce_op_OMP_optimized) {
  test_algorithm_2_omp_optimized alg_test(mr);
  X x{1, 1.0};

  // parallel execution
  double par_reduced = alg_test(*vec, x);
  EXPECT_EQ(par_reduced, expectedReduceResult);
}

TEST_P(CpuHostMemoryTest, Parallel_MapReduce_Lib_vs_op_OMP_Overhead) {
  test_algorithm_2 alg_lib(mr);
  test_algorithm_2_omp alg_test(mr);
  test_algorithm_2_omp_optimized alg_opt_test(mr);
  test_algorithm_2_seq alg_seq(mr);
  X x{1, 1.0};

  std::chrono::time_point<std::chrono::steady_clock> start_time;
  std::chrono::time_point<std::chrono::steady_clock> end_time;

  // start seq
  start_time = std::chrono::steady_clock::now();
  double seq = alg_seq(*vec, x);
  end_time = std::chrono::steady_clock::now();

  std::chrono::duration<double> diff_seq = end_time - start_time;
  std::cout << "Time for seq  = " << diff_seq.count() << " s\n";

  // start parallel but in seq mode map-reduce
  start_time = std::chrono::steady_clock::now();
  double *par_seq = alg_lib(*vec, x);
  end_time = std::chrono::steady_clock::now();

  std::chrono::duration<double> diff_par_seq = end_time - start_time;
  std::cout << "Time for lib seq  = " << diff_par_seq.count() << " s\n";

  // start lib map-reduce
  start_time = std::chrono::steady_clock::now();
  double reduced_lib = vecpar::omp::parallel_algorithm(alg_lib, mr, *vec, x);
  end_time = std::chrono::steady_clock::now();

  std::chrono::duration<double> diff_lib = end_time - start_time;
  std::cout << "Time for lib offload = " << diff_lib.count() << " s\n";

  // start OMP call operator
  start_time = std::chrono::steady_clock::now();
  double reduced_test = alg_test(*vec, x);
  end_time = std::chrono::steady_clock::now();

  std::chrono::duration<double> diff_test = end_time - start_time;
  std::cout << "Time for OMP code    = " << diff_test.count() << " s\n";

  // start OMP optimized implementation - call operator
  start_time = std::chrono::steady_clock::now();
  double reduced_opt_test = alg_opt_test(*vec, x);
  end_time = std::chrono::steady_clock::now();

  std::chrono::duration<double> diff_opt_test = end_time - start_time;
  std::cout << "Time for OMP opt code = " << diff_opt_test.count() << " s\n";

  // same result
  EXPECT_EQ(*par_seq, seq);
  EXPECT_EQ(reduced_lib, seq);
  EXPECT_EQ(reduced_lib, reduced_test);
  EXPECT_EQ(reduced_lib, reduced_opt_test);
}

TEST_P(CpuHostMemoryTest, Serial_MapFilter_MapReduce_Chained) {
  test_algorithm_3 first_alg(mr);
  test_algorithm_4 second_alg;

  vecmem::vector<double> first_result = first_alg(*vec);
  double *second_result = second_alg(first_result);

  EXPECT_EQ(*second_result, expectedFilterReduceResult);
}

TEST_P(CpuHostMemoryTest, Parallel_MapFilter_MapReduce_Chained) {
  test_algorithm_3 first_alg(mr);
  test_algorithm_4 second_alg;

  vecmem::vector<double> first_result =
      vecpar::omp::parallel_algorithm(first_alg, mr, *vec);
  double second_result =
      vecpar::omp::parallel_algorithm(second_alg, mr, first_result);

  EXPECT_EQ(second_result, expectedFilterReduceResult);
}

TEST_P(CpuHostMemoryTest, Parallel_MapFilter_MapReduce_Chained_With_Config) {
  test_algorithm_3 first_alg(mr);
  test_algorithm_4 second_alg;

  vecpar::config c{2, 5};
  vecmem::vector<double> first_result =
      vecpar::omp::parallel_algorithm(first_alg, mr, c, *vec);
  double second_result =
      vecpar::omp::parallel_algorithm(second_alg, mr, {1, 10}, first_result);

  EXPECT_EQ(second_result, expectedFilterReduceResult);
}

TEST_P(CpuHostMemoryTest, Parallel_Map_Extra_Param) {
  test_algorithm_5 alg;

  X x{1, 1.0};
  // parallel execution + destructive change on the input!!!
  vecmem::vector<double> result = vecpar::omp::parallel_map(alg, mr, *vec_d, x);
  EXPECT_EQ(result.size(), vec_d->size());
  for (int i = 0; i < result.size(); i++) {
    EXPECT_EQ(result.at(i), vec_d->at(i));
    EXPECT_EQ(result.at(i), (vec->at(i) + x.a) * x.b);
  }
}

INSTANTIATE_TEST_SUITE_P(OMP_HostMemory, CpuHostMemoryTest,
                         testing::ValuesIn(N));
} // namespace
