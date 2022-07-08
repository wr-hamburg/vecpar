#include <iostream>
#include <stdlib.h>

#include <vecmem/containers/vector.hpp>
#include <vecmem/memory/host_memory_resource.hpp>

#include "vecpar/omp/omp_parallelization.hpp"

#include "../../common/algorithm/test_algorithm_2.hpp"

#include "../../common/infrastructure/cleanup.hpp"
#include "native_algorithms/test_algorithm_2_omp.hpp"
#include "native_algorithms/test_algorithm_2_omp_optimized.hpp"
#include "native_algorithms/test_algorithm_2_seq.hpp"

vecmem::host_memory_resource mr;
test_algorithm_2 alg_lib(mr);
test_algorithm_2_omp alg_test(mr);
test_algorithm_2_omp_optimized alg_opt_test(mr);
test_algorithm_2_seq alg_seq(mr);
X x{1, 1.0};

std::chrono::time_point<std::chrono::steady_clock> start_time;
std::chrono::time_point<std::chrono::steady_clock> end_time;

void run_test_for_N(int n) {
  srand(time(NULL));
  int iSecret;

  vecmem::vector<int> *vec = new vecmem::vector<int>(n);
  double expectedReduceResult = 0;

  // init vector
  for (int i = 0; i < vec->size(); i++) {
    iSecret = rand() % 10 + 1;
    vec->at(i) = (iSecret % 2 == 0) ? i : (-i);
    expectedReduceResult += vec->at(i);
  }

  std::cout << "Test for N = " << vec->size() << std::endl;
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

  cleanup::free(*vec);
  std::cout << "***********************" << std::endl;
}

int main(int argc, char **argv) {
  std::vector<int> N = {10, 100, 133, 1000, 10000, 100000, 1000000};
  for (int i = 0; i < N.size(); i++) {
    run_test_for_N(N[i]);
  }
  return 0;
}
