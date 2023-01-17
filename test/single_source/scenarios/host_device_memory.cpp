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

#include "../../common/algorithm/test_algorithm_10.hpp"
#include "../../common/algorithm/test_algorithm_11.hpp"
#include "../../common/infrastructure/cleanup.hpp"
#include "vecpar/all/chain.hpp"
#include "vecpar/all/main.hpp"

namespace {

class SingleSourceHostDeviceMemoryTest
    : public TimeTest,
      public testing::WithParamInterface<int> {
public:
  SingleSourceHostDeviceMemoryTest() {
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

  ~SingleSourceHostDeviceMemoryTest() {
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

TEST_P(SingleSourceHostDeviceMemoryTest, Parallel_Map_Time) {
  std::chrono::time_point<std::chrono::steady_clock> start_time;
  std::chrono::time_point<std::chrono::steady_clock> end_time;

  test_algorithm_1 alg;

  start_time = std::chrono::steady_clock::now();
  vecpar::parallel_algorithm(alg, mr, *vec);
  end_time = std::chrono::steady_clock::now();

  std::chrono::duration<double> diff = end_time - start_time;
  printf("Parallel map time  = %f s\n", diff.count());
}

TEST_P(SingleSourceHostDeviceMemoryTest, Parallel_Map_Correctness) {
  test_algorithm_1 alg;
  vecmem::vector<double> result = vecpar::parallel_map(alg, mr, *vec);

  for (size_t i = 0; i < vec->size(); i++)
    EXPECT_EQ(vec->at(i) * 1.0, result.at(i));
}

TEST_P(SingleSourceHostDeviceMemoryTest, Parallel_Reduce_Time) {
  std::chrono::time_point<std::chrono::steady_clock> start_time;
  std::chrono::time_point<std::chrono::steady_clock> end_time;

  test_algorithm_1 alg;

  start_time = std::chrono::steady_clock::now();
  vecpar::parallel_reduce(alg, mr, *vec_d);
  end_time = std::chrono::steady_clock::now();

  std::chrono::duration<double> diff = end_time - start_time;
  printf("Parallel reduce <double> time  = %f s\n", diff.count());
}

TEST_P(SingleSourceHostDeviceMemoryTest, Parallel_Filter_Time) {
  std::chrono::time_point<std::chrono::steady_clock> start_time;
  std::chrono::time_point<std::chrono::steady_clock> end_time;

  test_algorithm_3 alg(mr);

  start_time = std::chrono::steady_clock::now();
  vecpar::parallel_filter(alg, mr, *vec_d);
  end_time = std::chrono::steady_clock::now();

  std::chrono::duration<double> diff = end_time - start_time;
  printf("Parallel filter time  = %f s\n", diff.count());
}

TEST_P(SingleSourceHostDeviceMemoryTest, Parallel_Filter_Correctness) {
  test_algorithm_3 alg(mr);

  vecmem::vector<double> result = vecpar::parallel_filter(alg, mr, *vec_d);

  int size = vec_d->size() % 2 == 0 ? int(vec_d->size() / 2)
                                    : int(vec_d->size() / 2) + 1;
  EXPECT_EQ(result.size(), size);

  // the order can be different
  std::sort(result.begin(), result.end());
  for (size_t i = 0; i < result.size(); i++) {
    EXPECT_EQ(vec_d->at(2 * i), result.at(i));
  }
}

TEST_P(SingleSourceHostDeviceMemoryTest, Parallel_MapReduce_Separately) {
  test_algorithm_1 alg;

  // parallel execution
  double result =
      vecpar::parallel_reduce(alg, mr, vecpar::parallel_map(alg, mr, *vec));

  EXPECT_EQ(result, expectedReduceResult);
}

TEST_P(SingleSourceHostDeviceMemoryTest, Parallel_MapReduce_Grouped) {
  test_algorithm_1 alg;

  // parallel execution
  double par_reduced = vecpar::parallel_algorithm(alg, mr, *vec);
  EXPECT_EQ(par_reduced, expectedReduceResult);
}

TEST_P(SingleSourceHostDeviceMemoryTest,
       Parallel_Extra_Params_MapReduce_Separately) {
  test_algorithm_2 alg;

  X x{1, 1.0};

  // parallel execution
  double result =
      vecpar::parallel_reduce(alg, mr, vecpar::parallel_map(alg, mr, *vec, x));

  EXPECT_EQ(result, expectedReduceResult);
}

TEST_P(SingleSourceHostDeviceMemoryTest,
       Parallel_Extra_Params_MapReduce_Grouped) {
  test_algorithm_2 alg;

  X x{1, 1.0};
  // parallel execution
  vecmem::vector<double> par_result(vec->size(), &mr);
  double par_reduced = vecpar::parallel_algorithm(alg, mr, *vec, x);
  EXPECT_EQ(par_reduced, expectedReduceResult);
}

TEST_P(SingleSourceHostDeviceMemoryTest, Parallel_MapFilter_MapReduce_Chained) {
  test_algorithm_3 first_alg(mr);
  test_algorithm_4 second_alg;

  double second_result = vecpar::parallel_algorithm(
      second_alg, mr, vecpar::parallel_algorithm(first_alg, mr, *vec));

  EXPECT_EQ(second_result, expectedFilterReduceResult);
}

TEST_P(SingleSourceHostDeviceMemoryTest, Parallel_Chained_two) {
  test_algorithm_3 first_alg(mr);
  test_algorithm_4 second_alg;

  //    vecpar::config c = {1, static_cast<int>(vec->size())};

  vecpar::chain<vecmem::host_memory_resource, double, vecmem::vector<int>>
      chain(mr);

  double second_result = chain //.with_config(c)
                             .with_algorithms(first_alg, second_alg)
                             .execute(*vec);

  EXPECT_EQ(second_result, expectedFilterReduceResult);
}

// mmap -> destructive change to input
TEST_P(SingleSourceHostDeviceMemoryTest, Parallel_Chained_one) {
  test_algorithm_5 first_alg;
  X x{1, 1.0};
  //    vecpar::config c = {1, static_cast<int>(vec->size())};

  vecpar::chain<vecmem::host_memory_resource, vecmem::vector<double>,
                vecmem::vector<double>, X>
      chain(mr);

  vecmem::vector<double> second_result = chain //.with_config(c)
                                             .with_algorithms(first_alg)
                                             .execute(*vec_d, x);

  for (size_t i = 0; i < second_result.size(); i++) {
    EXPECT_EQ(second_result[i], vec_d->at(i));
  }
}

// destructive test (will change vec_d)
TEST_P(SingleSourceHostDeviceMemoryTest, Parallel_MMap_Correctness) {
  test_algorithm_5 alg;
  X x{1, 1.0};
  vecmem::vector<double> result =
      vecpar::parallel_algorithm(alg, mr, *vec_d, x);

  for (size_t i = 0; i < vec_d->size(); i++)
    EXPECT_EQ(vec_d->at(i), result.at(i));
}

TEST_P(SingleSourceHostDeviceMemoryTest, Parallel_Map_Extra_Param) {
  test_algorithm_5 alg;

  X x{1, 1.0};
  // parallel execution + destructive change on the input!!!
  vecmem::vector<double> result =
      vecpar::parallel_algorithm(alg, mr, *vec_d, x);
  EXPECT_EQ(result.size(), vec_d->size());
  for (size_t i = 0; i < result.size(); i++) {
    EXPECT_EQ(result.at(i), vec_d->at(i));
    EXPECT_EQ(result.at(i), (vec->at(i) + x.a) * x.b);
  }
}

TEST_P(SingleSourceHostDeviceMemoryTest, Saxpy) {
  test_algorithm_6 alg;

  vecmem::vector<float> x(GetParam(), &mr);
  vecmem::vector<float> y(GetParam(), &mr);

  for (size_t i = 0; i < x.size(); i++) {
    x[i] = i;
    y[i] = 1.0;
  }
  float a = 5.0;

  vecmem::vector<float> result = vecpar::parallel_algorithm(alg, mr, y, x, a);

  for (size_t i = 0; i < result.size(); i++) {
    EXPECT_EQ(result.at(i), x[i] * a + 1.0);
  }

  cleanup::free(x);
  cleanup::free(y);
  cleanup::free(result);
}

TEST_P(SingleSourceHostDeviceMemoryTest, Saxpymzr) {
  test_algorithm_7 alg;

  vecmem::vector<double> x(GetParam(), &mr);
  vecmem::vector<int> y(GetParam(), &mr);
  vecmem::jagged_vector<float> z(GetParam(), &mr);

  double expected_result = 0.0;

  float a = 2.0;
  for (size_t i = 0; i < x.size(); i++) {
    x[i] = i;
    y[i] = 1;
    z[i].push_back(-1.0);
    // as map-reduce is implemented in algorithm 7
    expected_result += x[i] * a + y[i] * z[i][0];
  }

  double result = vecpar::parallel_algorithm(alg, mr, x, y, z, a);

  EXPECT_EQ(result, expected_result);

  cleanup::free(x);
  cleanup::free(y);
  cleanup::free(z);
}

TEST_P(SingleSourceHostDeviceMemoryTest, jagged_chained) {
  test_algorithm_10 first_alg;
  test_algorithm_11 second_alg;

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
      // map for alg10 and alg11
      expected[i].push_back(a * y[i][j] + x[i][j] - z[i] * t[i] * v[i][j] +
                            5.0);
    }
  }

  vecpar::chain<vecmem::host_memory_resource,
                // result for the entire chain:
                vecmem::jagged_vector<double>,
                // input for first algorithm: (5 collections + 1 extra arg)
                vecmem::jagged_vector<double>, vecmem::jagged_vector<double>,
                vecmem::vector<int>, vecmem::vector<int>,
                vecmem::jagged_vector<int>, double>
      this_chain(mr);

  this_chain.with_algorithms(first_alg, second_alg).execute(x, y, z, t, v, a);

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

INSTANTIATE_TEST_SUITE_P(HostDeviceMemory, SingleSourceHostDeviceMemoryTest,
                         testing::ValuesIn(N));
} // namespace
