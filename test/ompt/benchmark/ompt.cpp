#include <iostream>
#include <stdlib.h>

#include "vecpar/ompt/ompt_parallelization.hpp"
#include <string>
#include <vecmem/containers/vector.hpp>
#include <vecmem/memory/host_memory_resource.hpp>

#include "omp.h"

#include "../../common/algorithm/test_algorithm_2.hpp"
#include "../../common/infrastructure/TimeLogger.hpp"

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
  for (int i = 0; i < vec->size(); i++) {
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
  std::cout << "Time for lib offload map = " << diff_map.count() << " s\n";
  std::chrono::duration<double> diff_reduce = end_time - middle_time;
  std::cout << "Time for lib offload reduce = " << diff_reduce.count()
            << " s\n";
  std::chrono::duration<double> diff_lib = end_time - start_time;
  std::cout << "Time for lib offload = " << diff_lib.count() << " s\n";

  
  //std::cout << *par_seq << "; " << par_lib << std::endl;
  assert(*par_seq == par_lib);
  
  write_to_csv(filename, vec->size(), diff_seq.count(), diff_map.count(), diff_reduce.count(), diff_lib.count());
}

int main(int argc, char **argv) {

#if defined(COMPILE_FOR_HOST)
  std::string filename = "benchmark_ompt_cpu.csv";
#else
  std::string filename = "benchmark_ompt_gpu.csv";
#endif

  printf("%s\n", filename.c_str());

  write_to_csv(filename, "N", "sequential", "lib_map", "lib_reduce", "lib_total");

  std::vector<int> N = {10, 1000, 100000, 1000000, 10000000};
  for (int i = 0; i < N.size(); i++) {
    run_test_for_N(N[i], filename);
  }
  return 0;
}
