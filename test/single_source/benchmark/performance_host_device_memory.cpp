#include <gtest/gtest.h>

#include <vecmem/containers/vector.hpp>
#include <vecmem/memory/host_memory_resource.hpp>

#include "../../common/infrastructure/TimeLogger.hpp"
#include "../../common/infrastructure/TimeTest.hpp"
#include "../../common/infrastructure/sizes.hpp"

#include "../../common/algorithm/test_algorithm_3.hpp"
#include "../../common/algorithm/test_algorithm_4.hpp"
#include "../../common/algorithm/test_algorithm_6.hpp"

#include "../../common/infrastructure/cleanup.hpp"
#include "vecpar/all/chain.hpp"
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
};

TEST_P(PerformanceTest_HostDevice, Chain) {
  test_algorithm_3 first_alg(mr);
  test_algorithm_4 second_alg;

  std::chrono::time_point<std::chrono::steady_clock> start_time;
  std::chrono::time_point<std::chrono::steady_clock> end_time;

  start_time = std::chrono::steady_clock::now();
  vecpar::parallel_algorithm(second_alg, mr,
                             vecpar::parallel_algorithm(first_alg, mr, *vec));
  end_time = std::chrono::steady_clock::now();

  std::chrono::duration<double> diff1 = end_time - start_time;
  printf("Default = %f s\n", diff1.count());

  start_time = std::chrono::steady_clock::now();
  vecpar::chain<vecmem::host_memory_resource, double, vecmem::vector<int>>
      chain(mr);

  chain //.with_config(c)
      .with_algorithms(first_alg, second_alg)
      .execute(*vec);
  end_time = std::chrono::steady_clock::now();

  std::chrono::duration<double> diff2 = end_time - start_time;
  printf("Chain  = %f s\n", diff2.count());

#if defined(__CUDA__) && defined(__clang__)
  write_to_csv("gpu_chain_hd.csv", GetParam(), diff1.count(), diff2.count());
#else
  write_to_csv("cpu_chain_hd.csv", GetParam(), diff1.count(), diff2.count());
#endif
}

TEST_P(PerformanceTest_HostDevice, Saxpy) {
  test_algorithm_6 alg;

  vecmem::vector<float> x(GetParam(), &mr);
  vecmem::vector<float> y(GetParam(), &mr);

  for (int i = 0; i < GetParam(); i++) {
    x[i] = i;
    y[i] = i - 1;
  }
  float a = 2.0;

  std::chrono::time_point<std::chrono::steady_clock> start_time;
  std::chrono::time_point<std::chrono::steady_clock> end_time;

  start_time = std::chrono::steady_clock::now();
  vecpar::parallel_algorithm(alg, mr, y, x, a);
  end_time = std::chrono::steady_clock::now();

  for (size_t i = 0; i < y.size(); i++) {
    EXPECT_EQ(y[i], x[i] * a + (x[i] - 1));
  }
  std::chrono::duration<double> diff = end_time - start_time;
  printf("SAXPY mmap time  = %f s\n", diff.count());

#if defined(__CUDA__) && defined(__clang__)
  write_to_csv("gpu_saxpy_hd.csv", GetParam(), diff.count());
#else
  write_to_csv("cpu_saxpy_hd.csv", GetParam(), diff.count());
#endif

  cleanup::free(x);
  cleanup::free(y);
}

INSTANTIATE_TEST_SUITE_P(PerformanceTest_HostDevice, PerformanceTest_HostDevice,
                         testing::ValuesIn(N));
} // end namespace