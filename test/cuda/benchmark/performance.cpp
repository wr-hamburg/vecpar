#include <gtest/gtest.h>

#include "../../common/infrastructure/TimeTest.hpp"
#include "../../common/infrastructure/cleanup.hpp"
#include "../../common/infrastructure/sizes.hpp"
#include <vecmem/memory/host_memory_resource.hpp>

#include "../../common/algorithm/test_algorithm_2.hpp"
#include "native_algorithms/test_algorithm_2_cuda_hm.hpp"
#include "native_algorithms/test_algorithm_2_cuda_mm.hpp"
#include "vecpar/cuda/cuda_parallelization.hpp"

namespace {
class CUDA_PerformanceTest : public TimeTest,
                             public testing::WithParamInterface<int> {
public:
  CUDA_PerformanceTest() {
    mm_vec = new vecmem::vector<int>(GetParam(), &mm);
    hm_vec = new vecmem::vector<int>(GetParam(), &hm);

    for (int i = 0; i < GetParam(); i++) {
      mm_vec->at(i) = i;
      hm_vec->at(i) = i;
      expectedReduceResult += i * 1.0;
    }
    printf("*******************************\n");
  }

  ~CUDA_PerformanceTest() {
    cleanup::free(*mm_vec);
    cleanup::free(*hm_vec);
  }

protected:
  vecmem::host_memory_resource hm;
  vecmem::cuda::managed_memory_resource mm;
  vecmem::vector<int> *mm_vec;
  vecmem::vector<int> *hm_vec;
  double expectedReduceResult = 0;
};

TEST_P(CUDA_PerformanceTest, Lib_Overhead_ManagedMemory) {
  test_algorithm_2 alg(mm);
  test_algorithm_2_cuda_mm alg_cuda(mm);

  X x{1, 1.0};

  std::chrono::time_point<std::chrono::steady_clock> start_time;
  std::chrono::time_point<std::chrono::steady_clock> end_time;

  start_time = std::chrono::steady_clock::now();
  double par_reduced = vecpar::cuda::parallel_algorithm(alg, mm, *mm_vec, x);
  end_time = std::chrono::steady_clock::now();

  std::chrono::duration<double> diff = end_time - start_time;
  std::cout << "Time for MapReduce vecpar  = " << diff.count() << " s\n";

  start_time = std::chrono::steady_clock::now();
  double reduced = alg_cuda(*mm_vec, x);
  end_time = std::chrono::steady_clock::now();

  diff = end_time - start_time;
  std::cout << "Time for CUDA              = " << diff.count() << " s\n";
  EXPECT_EQ(par_reduced, reduced);
}

TEST_P(CUDA_PerformanceTest, Lib_Overhead_HostDeviceMemory) {
  test_algorithm_2 alg(hm);
  test_algorithm_2_cuda_hm alg_cuda(hm);

  X x{1, 1.0};

  std::chrono::time_point<std::chrono::steady_clock> start_time;
  std::chrono::time_point<std::chrono::steady_clock> end_time;

  start_time = std::chrono::steady_clock::now();
  double par_reduced = vecpar::cuda::parallel_algorithm(alg, hm, *hm_vec, x);
  end_time = std::chrono::steady_clock::now();

  std::chrono::duration<double> diff = end_time - start_time;
  std::cout << "Time for MapReduce vecpar  = " << diff.count() << " s\n";

  start_time = std::chrono::steady_clock::now();
  double reduced = alg_cuda(*hm_vec, x);
  end_time = std::chrono::steady_clock::now();

  diff = end_time - start_time;
  std::cout << "Time for CUDA              = " << diff.count() << " s\n";
  EXPECT_EQ(par_reduced, reduced);
}

INSTANTIATE_TEST_SUITE_P(CUDA_HostDevice, CUDA_PerformanceTest,
                         testing::ValuesIn(N));
} // end namespace