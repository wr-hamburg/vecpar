#ifndef VECPAR_OMP_INTERNAL_HPP
#define VECPAR_OMP_INTERNAL_HPP

#include "config.hpp"
#include <omp.h>

namespace internal {

template <typename Function, typename... Arguments>
void offload_map(vecpar::config config, int size, Function f,
                 Arguments... args) {
  int threadsNum;

  if (config.isEmpty()) {
#pragma omp parallel for
    for (int i = 0; i < size; i++) {
      f(i, args...);
      DEBUG_ACTION(threadsNum = omp_get_num_threads();)
    }
  } else {
#pragma omp parallel for num_threads(config.m_gridSize *config.m_blockSize)
    for (int i = 0; i < size; i++) {
      f(i, args...);
      DEBUG_ACTION(threadsNum = omp_get_num_threads();)
    }
  }
  DEBUG_ACTION(printf("Using %d OpenMP threads \n", threadsNum);)
}

/// based on article:
/// https://coderwall.com/p/gocbhg/openmp-improve-reduction-techniques
template <typename R, typename Function>
void offload_reduce(int size, R *result, Function f,
                    vecmem::vector<R> &map_result) {
#pragma omp parallel
  {
    R *tmp_result = new R();
#pragma omp for nowait
    for (int i = 0; i < size; i++)
      f(tmp_result, map_result[i]);

#pragma omp critical
    f(result, *tmp_result);
  }
}

template <typename R, typename Function, typename... Arguments>
void offload_filter(int size, R *result, Function f, Arguments... args) {
  int idx = 0;
#pragma omp teams distribute parallel for
  for (int i = 0; i < size; i++) {
#pragma omp critical
    f(i, idx, *result, args...);
  }
  result->resize(idx);
}
} // namespace internal
#endif // VECPAR_OMP_INTERNAL_HPP
