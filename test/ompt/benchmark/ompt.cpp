#include <cstdio>
#include <iostream>
#include <stdlib.h>

#include <string>
#include <vecmem/containers/vector.hpp>
#include <vecmem/memory/host_memory_resource.hpp>

#include "omp.h"

#include "../../common/algorithm/test_algorithm_2.hpp"
#include "../../common/infrastructure/TimeLogger.hpp"
#include "vecpar/ompt/ompt_parallelization.hpp"

vecmem::host_memory_resource mr;
std::chrono::time_point<std::chrono::steady_clock> start_time;
std::chrono::time_point<std::chrono::steady_clock> middle_time;
std::chrono::time_point<std::chrono::steady_clock> end_time;

test_algorithm_2 alg_lib;
X x{1, 1.0};

void run_test_for_N(int n, std::string filename) {

  srand(time(NULL));
  int iSecret;

  vecmem::vector<int> *vec = new vecmem::vector<int>(n);
  double expectedReduceResult = 0;

  // init vector
  for (std::size_t i = 0; i < vec->size(); i++) {
    iSecret = rand() % 10 + 1;
    vec->at(i) = (iSecret % 2 == 0) ? i : (-i);
    //std::cout << vec->at(i) << std::endl;
    expectedReduceResult += vec->at(i) > 0 ? vec->at(i): 0;
  }

  //std::cout << expectedReduceResult << std::endl;

  std::cout << "Test for N = " << vec->size() << std::endl;

  // start seq
  start_time = std::chrono::steady_clock::now();
  double *par_seq = alg_lib(*vec, x);
  end_time = std::chrono::steady_clock::now();

  std::chrono::duration<double> diff_seq = end_time - start_time;
  std::cout << "Time for seq  = " << diff_seq.count() << " s\n";

  // start lib map-reduce
  start_time = std::chrono::steady_clock::now();

  auto map_result = vecpar::ompt::parallel_map(alg_lib, mr, *vec, x);
  middle_time = std::chrono::steady_clock::now();
  double par_lib = vecpar::ompt::parallel_reduce(alg_lib, mr, map_result);
  
  end_time = std::chrono::steady_clock::now();

  std::chrono::duration<double> diff_map = middle_time - start_time;
  std::cout << "Time for lib seperate offload map = " << diff_map.count() << " s\n";
  std::chrono::duration<double> diff_reduce = end_time - middle_time;
  std::cout << "Time for lib seperate offload reduce = " << diff_reduce.count()
            << " s\n";
  std::chrono::duration<double> diff_lib = end_time - start_time;
  std::cout << "Time for lib seperate offload = " << diff_lib.count() << " s\n";

  

  start_time = std::chrono::steady_clock::now();
  double par_lib_grouped = vecpar::ompt::parallel_algorithm(alg_lib, mr, *vec, x);
  end_time = std::chrono::steady_clock::now();

  std::chrono::duration<double> diff_lib_grouped = end_time - start_time;
  std::cout << "Time for lib grouped offload = " << diff_lib_grouped.count() << " s\n";
  
  std::cout << expectedReduceResult << "; " << *par_seq << "; " << par_lib << "; " << par_lib_grouped << std::endl;

#if 1
  assert(expectedReduceResult == *par_seq);
  assert(expectedReduceResult == par_lib);
  assert(expectedReduceResult == par_lib_grouped);
#endif
  
  write_to_csv(filename, vec->size(), diff_seq.count(), diff_map.count(), diff_reduce.count(), diff_lib.count(), diff_lib_grouped.count());
}

int main() {

#if defined(COMPILE_FOR_HOST)
  std::string filename = "benchmark_ompt_cpu.csv";
#else
  std::string filename = "benchmark_ompt_gpu.csv";
#endif

  write_to_csv(filename, "N", "sequential", "lib_map", "lib_reduce", "lib_total", "lib_grouped");

    std::vector<int> N = {10, 1000, 100000, 1000000, 10000000};
  for (std::size_t i = 0; i < N.size(); i++) {
    run_test_for_N(N[i], filename);
  }
  return 0;
}
