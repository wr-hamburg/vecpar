#include <iostream>
#include <vecmem/memory/host_memory_resource.hpp>

#include "../../common/algorithm/test_algorithm_2.hpp"
#include "../../common/infrastructure/cleanup.hpp"
#include "native_algorithms/test_algorithm_2_cuda_hm.hpp"
#include "native_algorithms/test_algorithm_2_cuda_mm.hpp"
#include "vecpar/cuda/cuda_parallelization.hpp"

std::chrono::time_point<std::chrono::steady_clock> start_time;
std::chrono::time_point<std::chrono::steady_clock> end_time;

void run_test_for_N(vecmem::host_memory_resource mem, int n) {

  double expectedReduceResult = 0;
  vecmem::vector<int> *vec = new vecmem::vector<int>(n, &mem);

  for (int i = 0; i < n; i++) {
    vec->at(i) = i;
    expectedReduceResult += i * 1.0;
  }

  std::cout << "Test for N = " << vec->size() << std::endl;

  test_algorithm_2 alg;
  test_algorithm_2_cuda_hm alg_cuda(mem);
  X x{1, 1.0};

  start_time = std::chrono::steady_clock::now();
  double par_reduced = vecpar::cuda::parallel_algorithm(alg, mem, *vec, x);
  end_time = std::chrono::steady_clock::now();

  std::chrono::duration<double> diff = end_time - start_time;
  std::cout << "Time for MapReduce vecpar  = " << diff.count() << " s\n";

  start_time = std::chrono::steady_clock::now();
  double reduced = alg_cuda(*vec, x);
  end_time = std::chrono::steady_clock::now();

  diff = end_time - start_time;
  std::cout << "Time for CUDA              = " << diff.count() << " s\n";

  cleanup::free(*vec);
  std::cout << "***********************" << std::endl;
}

void run_test_for_N(vecmem::cuda::managed_memory_resource mem, int n) {

  double expectedReduceResult = 0;
  vecmem::vector<int> *vec = new vecmem::vector<int>(n, &mem);

  for (int i = 0; i < n; i++) {
    vec->at(i) = i;
    expectedReduceResult += i * 1.0;
  }

  std::cout << "Test for N = " << vec->size() << std::endl;

  test_algorithm_2 alg;
  test_algorithm_2_cuda_mm alg_cuda(mem);
  X x{1, 1.0};

  start_time = std::chrono::steady_clock::now();
  double par_reduced = vecpar::cuda::parallel_algorithm(alg, mem, *vec, x);
  end_time = std::chrono::steady_clock::now();

  std::chrono::duration<double> diff = end_time - start_time;
  std::cout << "Time for MapReduce vecpar  = " << diff.count() << " s\n";

  start_time = std::chrono::steady_clock::now();
  double reduced = alg_cuda(*vec, x);
  end_time = std::chrono::steady_clock::now();

  diff = end_time - start_time;
  std::cout << "Time for CUDA              = " << diff.count() << " s\n";

  cleanup::free(*vec);
  std::cout << "***********************" << std::endl;
}

int main(int argc, char **argv) {
  std::vector<int> N = {10, 1000, 100000, 1000000, 10000000};
  std::cout << "Tests using managed memory: " << std::endl;
  vecmem::cuda::managed_memory_resource mm;
  for (int i = 0; i < N.size(); i++) {
    run_test_for_N(mm, N[i]);
  }

  std::cout << "Tests using host-device memory: " << std::endl;
  vecmem::host_memory_resource hm;
  for (int i = 0; i < N.size(); i++) {
    run_test_for_N(hm, N[i]);
  }
  return 0;
}
