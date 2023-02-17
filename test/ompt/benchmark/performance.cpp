#include <gtest/gtest.h>

#include <vecmem/containers/vector.hpp>
#include <vecmem/memory/host_memory_resource.hpp>

#include "../../common/infrastructure/TimeLogger.hpp"
#include "../../common/infrastructure/TimeTest.hpp"
#include "../../common/infrastructure/sizes.hpp"

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

TEST_P(PerformanceTest_HostDevice, Saxpy) {
  saxpy alg;

  vecmem::vector<float> *x = new vecmem::vector<float>(GetParam(), &mr);
  vecmem::vector<float> *y = new vecmem::vector<float>(GetParam(), &mr);
  vecmem::vector<float> *expected_result =
      new vecmem::vector<float>(GetParam(), &mr);
  float a = 2.0;

  // init vec
  for (int i = 0; i < GetParam(); i++) {
    x->at(i) = i % 100;
    y->at(i) = (i - 1) % 100;
    expected_result->at(i) = y->at(i) + x->at(i) * a;
  }

  start_time = std::chrono::steady_clock::now();
  vecpar::ompt::parallel_algorithm(alg, mr, *y, *x, a);
  end_time = std::chrono::steady_clock::now();
  // check results
  for (size_t i = 0; i < y->size(); i++) {
    EXPECT_EQ(y->at(i), expected_result->at(i));
  }
  // check time
  std::chrono::duration<double> diff = end_time - start_time;
  printf("SAXPY vecpar OMPT time  = %f s\n", diff.count());

#if defined(COMPILE_FOR_DEVICE)
  write_to_csv("gpu_saxpy_ompt_hd.csv", GetParam(), diff.count());
#else
  write_to_csv("cpu_saxpy_ompt_hd.csv", GetParam(), diff.count());
#endif

  cleanup::free(*x);
  cleanup::free(*y);
}

TEST_P(PerformanceTest_HostDevice, Daxpy) {
  daxpy alg;

  vecmem::vector<double> *x = new vecmem::vector<double>(GetParam(), &mr);
  vecmem::vector<double> *y = new vecmem::vector<double>(GetParam(), &mr);
  vecmem::vector<double> *expected_result =
      new vecmem::vector<double>(GetParam(), &mr);
  double a = 2.0;

  // init vec
  for (int i = 0; i < GetParam(); i++) {
    x->at(i) = i % 100;
    y->at(i) = (i - 1) % 100;
    expected_result->at(i) = y->at(i) + x->at(i) * a;
  }

  start_time = std::chrono::steady_clock::now();
  vecpar::ompt::parallel_algorithm(alg, mr, *y, *x, a);
  end_time = std::chrono::steady_clock::now();
  // check results
  for (size_t i = 0; i < y->size(); i++) {
    EXPECT_EQ(y->at(i), expected_result->at(i));
  }
  // check time
  std::chrono::duration<double> diff = end_time - start_time;
  printf("DAXPY vecpar time  = %f s\n", diff.count());

#if defined(COMPILE_FOR_DEVICE)
  write_to_csv("gpu_daxpy_ompt_hd.csv", GetParam(), diff.count());
#else
  write_to_csv("cpu_daxpy_ompt_hd.csv", GetParam(), diff.count());
#endif

  cleanup::free(*x);
  cleanup::free(*y);
}

INSTANTIATE_TEST_SUITE_P(PerformanceTest_HostDevice, PerformanceTest_HostDevice,
                         testing::ValuesIn(N));
} // namespace
