#include <gtest/gtest.h>

#include <vecmem/containers/jagged_vector.hpp>
#include <vecmem/containers/vector.hpp>
#include <vecmem/memory/host_memory_resource.hpp>

#include "../../common/infrastructure/TimeTest.hpp"
#include "../../common/infrastructure/sizes.hpp"

#include "../../common/algorithm/test_algorithm_1.hpp"
#include "../../common/algorithm/test_algorithm_2.hpp"
#include "../../common/algorithm/test_algorithm_3.hpp"
#include "../../common/algorithm/test_algorithm_4.hpp"
#include "../../common/algorithm/test_algorithm_5.hpp"
#include "../../common/algorithm/test_algorithm_6.hpp"
#include "../../common/algorithm/test_algorithm_7.hpp"
#include "../../common/algorithm/test_algorithm_8.hpp"

#include "../../common/algorithm/test_algorithm_10.hpp"
#include "../../common/algorithm/test_algorithm_9.hpp"
#include "../../common/infrastructure/cleanup.hpp"
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

  ~CpuHostMemoryTest() {
    cleanup::free(*vec);
    cleanup::free(*vec_d);
  }

protected:
  vecmem::host_memory_resource mr;
  vecmem::vector<int> *vec;
  vecmem::vector<double> *vec_d;
  double expectedReduceResult = 0;
  double expectedFilterReduceResult = 0;
};

TEST_P(CpuHostMemoryTest, Parallel_MapOnly) {
  test_algorithm_1 alg;

  vecmem::vector<double> par_result(vec->size(), &mr);

  vecpar::omp::parallel_map(vec->size(), [&] (int idx)  {
    alg.mapping_function(par_result[idx], vec->at(idx));
  });

  EXPECT_EQ(par_result[0], vec->at(0));
  EXPECT_EQ(par_result[int(GetParam() / 2)], vec->at(int(GetParam() / 2)));
  EXPECT_EQ(par_result[GetParam() - 1], vec->at(GetParam() - 1));
}

TEST_P(CpuHostMemoryTest, Parallel_ReduceOnly) {
  test_algorithm_1 alg;

  double *result = new double();
  vecpar::omp::parallel_reduce(
      vec->size(), result,
      [&] (double *r, double tmp)  { alg.reducing_function(r, tmp); },
      [&] () {return 0.0;},
      *vec_d);

  EXPECT_EQ(*result, expectedReduceResult);
}

TEST_P(CpuHostMemoryTest, Parallel_Inline_lambda) {
  test_algorithm_1 alg;

  vecpar::omp::parallel_map(
      vec->size(), [&] (int idx)  { vec->at(idx) *= 4.0; });

  EXPECT_EQ(vec->at(0), 0);
  EXPECT_EQ(vec->at(1), 4.);
  EXPECT_EQ(vec->at(2), 8.);
}

TEST_P(CpuHostMemoryTest, Parallel_Map_Time) {
  std::chrono::time_point<std::chrono::steady_clock> start_time;
  std::chrono::time_point<std::chrono::steady_clock> end_time;

  test_algorithm_1 alg;

  start_time = std::chrono::steady_clock::now();
  vecmem::vector<double> result = vecpar::omp::parallel_map(alg, mr, *vec);
  end_time = std::chrono::steady_clock::now();

  std::chrono::duration<double> diff = end_time - start_time;
  printf("Parallel map time  = %f s\n", diff.count());
}

TEST_P(CpuHostMemoryTest, Parallel_Map_Correctness) {
  test_algorithm_1 alg;
  vecmem::vector<double> result = vecpar::omp::parallel_map(alg, mr, *vec);

  for (int i = 0; i < vec->size(); i++)
    EXPECT_EQ(vec->at(i) * 1.0, result.at(i));
}

TEST_P(CpuHostMemoryTest, Parallel_Reduce_Time) {
  std::chrono::time_point<std::chrono::steady_clock> start_time;
  std::chrono::time_point<std::chrono::steady_clock> end_time;

  test_algorithm_1 alg;

  start_time = std::chrono::steady_clock::now();
  double result = vecpar::omp::parallel_reduce(alg, mr, *vec_d);
  end_time = std::chrono::steady_clock::now();

  std::chrono::duration<double> diff = end_time - start_time;
  printf("Parallel reduce <double> time  = %f s\n", diff.count());
}

TEST_P(CpuHostMemoryTest, Parallel_Reduce_Correctness) {
  test_algorithm_1 alg;
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
  test_algorithm_1 alg;

  // serial execution
  double *result = alg(*vec);
  EXPECT_EQ(*result, expectedReduceResult);
}

TEST_P(CpuHostMemoryTest, Parallel_MapReduce_Separately) {
  test_algorithm_1 alg;

  // parallel execution
  vecmem::vector<double> par_result(vec->size(), &mr);

  vecpar::omp::parallel_map(vec->size(), [&] (int idx)  {
    alg.mapping_function(par_result[idx], vec->at(idx));
  });

  double *result = new double();
  vecpar::omp::parallel_reduce(
      vec->size(), result,
      [&] (double *r, double tmp)  { alg.reducing_function(r, tmp); },
      [&] () {return 0.0;},
      par_result);

  EXPECT_EQ(*result, expectedReduceResult);
}

TEST_P(CpuHostMemoryTest, Parallel_MapReduce_Grouped) {
  test_algorithm_1 alg;

  // parallel execution
  double par_reduced = vecpar::omp::parallel_algorithm(alg, mr, *vec);
  EXPECT_EQ(par_reduced, expectedReduceResult);
}

TEST_P(CpuHostMemoryTest, Serial_Extra_Params_MapReduce) {
  test_algorithm_2 alg;
  X x{1, 1.0};

  // serial execution
  double *result = alg(*vec, x);
  EXPECT_EQ(*result, expectedReduceResult);
}

TEST_P(CpuHostMemoryTest, Parallel_Extra_Params_MapReduce_Separately) {
  test_algorithm_2 alg;

  X x{1, 1.0};

  // parallel execution
  vecmem::vector<double> par_result(vec->size(), &mr);

  vecpar::omp::parallel_map(vec->size(), [&] (int idx)  {
    alg.mapping_function(par_result[idx], vec->at(idx), x);
  });

  double *result = new double();
  vecpar::omp::parallel_reduce(
      vec->size(), result,
      [&] (double *r, double tmp)  { alg.reducing_function(r, tmp); },
      [&] () {return 0.0;},
      par_result);

  EXPECT_EQ(*result, expectedReduceResult);
}

TEST_P(CpuHostMemoryTest, Parallel_Extra_Params_MapReduce_Grouped) {
  test_algorithm_2 alg;

  X x{1, 1.0};
  // parallel execution
  double par_reduced = vecpar::omp::parallel_algorithm(alg, mr, *vec, x);
  EXPECT_EQ(par_reduced, expectedReduceResult);
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
  vecpar::config c(2,5);
  // parallel execution + destructive change on the input!!!
  vecmem::vector<double> result = vecpar::omp::parallel_map(alg, mr, c, *vec_d, x);
  EXPECT_EQ(result.size(), vec_d->size());
  for (int i = 0; i < result.size(); i++) {
    EXPECT_EQ(result.at(i), vec_d->at(i));
    EXPECT_EQ(result.at(i), (vec->at(i) + x.a) * x.b);
  }
}

TEST_P(CpuHostMemoryTest, two_collections) {
  test_algorithm_6 alg;

  vecmem::vector<float> x(GetParam(), &mr);
  vecmem::vector<float> y(GetParam(), &mr);

  for (int i = 0; i < x.size(); i++) {
    x[i] = i;
    y[i] = 1.0;
  }
  float a = 2.0;
  vecmem::vector<float> result = vecpar::omp::parallel_map(alg, mr, y, x, a);

  for (int i = 0; i < result.size(); i++) {
    EXPECT_EQ(result.at(i), x[i] * a + 1.0);
  }

  cleanup::free(x);
  cleanup::free(y);
  cleanup::free(result);
}

TEST_P(CpuHostMemoryTest, three_collections) {
  test_algorithm_7 alg;

  vecmem::vector<double> x(GetParam(), &mr);
  vecmem::vector<int> y(GetParam(), &mr);
  vecmem::jagged_vector<float> z(GetParam(), &mr);

  double expected_result = 0.0;

  float a = 2.0;
  for (int i = 0; i < x.size(); i++) {
    x[i] = i;
    y[i] = 1;
    z[i].push_back(-1.0);
    // as map-reduce is implemented in algorithm 7
    expected_result += x[i] * a + y[i] * z[i][0];
  }

  double result = vecpar::omp::parallel_algorithm(alg, mr, x, y, z, a);
  end_time = std::chrono::steady_clock::now();

  EXPECT_EQ(result, expected_result);

  cleanup::free(x);
  cleanup::free(y);
  cleanup::free(z);
}

TEST_P(CpuHostMemoryTest, four_collections) {
  test_algorithm_8 alg;

  vecmem::vector<double> x(GetParam(), &mr);
  vecmem::vector<int> y(GetParam(), &mr);
  vecmem::vector<float> z(GetParam(), &mr);
  vecmem::jagged_vector<float> t(GetParam(), &mr);

  vecmem::vector<double> expected;

  float a = 2.0;
  for (int i = 0; i < x.size(); i++) {
    x[i] = i;
    y[i] = 1;
    z[i] = -1.0;
    t[i].push_back(4.0 * i);
    // as map is implemented in algorithm 8
    double tmp = x[i] * a + y[i] * z[i] * t[i][0];
    // as filter is implemented in algorithm 8
    if (tmp < 0)
      expected.push_back(tmp);
  }

  vecmem::vector<double> result =
      vecpar::omp::parallel_algorithm(alg, mr, x, y, z, t, a);

  EXPECT_EQ(result.size(), expected.size());

  // the result can be in a different order
  std::sort(expected.begin(), expected.end());
  std::sort(result.begin(), result.end());
  for (int i = 0; i < result.size(); i++) {
    EXPECT_EQ(result.at(i), expected.at(i));
  }

  cleanup::free(x);
  cleanup::free(y);
  cleanup::free(z);
  cleanup::free(t);
  cleanup::free(expected);
  cleanup::free(result);
}

TEST_P(CpuHostMemoryTest, five_collections) {
  test_algorithm_9 alg;

  vecmem::vector<double> x(GetParam(), &mr);
  vecmem::vector<int> y(GetParam(), &mr);
  vecmem::vector<float> z(GetParam(), &mr);
  vecmem::vector<float> t(GetParam(), &mr);
  vecmem::vector<int> v(GetParam(), &mr);

  vecmem::vector<double> expected;

  float a = 2.0;
  for (int i = 0; i < x.size(); i++) {
    x[i] = i;
    y[i] = 1;
    z[i] = -1.0;
    t[i] = 4.0 * i;
    v[i] = i - 1;
    // as map is implemented in algorithm 8
    double tmp = x[i] * a + y[i] * z[i] * t[i] + v[i];
    // as filter is implemented in algorithm 8
    if (tmp > 0)
      expected.push_back(tmp);
  }

  vecmem::vector<double> result =
      vecpar::omp::parallel_algorithm(alg, mr, x, y, z, t, v, a);

  EXPECT_EQ(result.size(), expected.size());

  // the result can be in a different order
  std::sort(expected.begin(), expected.end());
  std::sort(result.begin(), result.end());
  for (int i = 0; i < result.size(); i++) {
    EXPECT_EQ(result.at(i), expected.at(i));
  }

  cleanup::free(x);
  cleanup::free(y);
  cleanup::free(z);
  cleanup::free(t);
  cleanup::free(v);
  cleanup::free(expected);
  cleanup::free(result);
}

TEST_P(CpuHostMemoryTest, five_jagged) {
  std::chrono::time_point<std::chrono::steady_clock> start_time;
  std::chrono::time_point<std::chrono::steady_clock> end_time;

  test_algorithm_10 alg;

  vecmem::jagged_vector<double> x(GetParam(), &mr);
  vecmem::jagged_vector<double> y(GetParam(), &mr);
  vecmem::vector<int> z(GetParam(), &mr);
  vecmem::vector<int> t(GetParam(), &mr);
  vecmem::jagged_vector<int> v(GetParam(), &mr);

  double a = 2.0;

  // make sure the 2d collection is now square and it is
  // small enough
  int N = 10; // second dimension
  vecmem::jagged_vector<double> expected(GetParam(), &mr);
  for (int i = 0; i < GetParam(); i++) {
    z[i] = -i;
    t[i] = -2;
    for (int j = 0; j < N; j++) {
      x[i].push_back(1);
      y[i].push_back(i);
      v[i].push_back(10);
      expected[i].push_back(a * y[i][j] + x[i][j] - z[i] * t[i] * v[i][j]);
    }
  }

  start_time = std::chrono::steady_clock::now();
  vecpar::omp::parallel_map(alg, mr, x, y, z, t, v, a);
  end_time = std::chrono::steady_clock::now();

  std::chrono::duration<double> diff = end_time - start_time;
  printf("Parallel map time  = %f s\n", diff.count());

  for (int i = 0; i < GetParam(); i++) {
    for (int j = 0; j < N; j++) {
      EXPECT_EQ(x[i][j], expected[i][j]);
    }
  }

  cleanup::free(x);
  cleanup::free(y);
  cleanup::free(z);
  cleanup::free(t);
  cleanup::free(v);
  cleanup::free(expected);
}

INSTANTIATE_TEST_SUITE_P(Trivial_HostMemory, CpuHostMemoryTest,
                         testing::ValuesIn(N));
} // namespace
