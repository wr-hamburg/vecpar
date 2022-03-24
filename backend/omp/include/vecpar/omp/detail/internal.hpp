#ifndef VECPAR_OMP_INTERNAL_HPP
#define VECPAR_OMP_INTERNAL_HPP

#include <omp.h>
#include "config.hpp"

namespace internal {
    template<typename Function, typename... Arguments>
    void offload_map(vecpar::config config, int size, Function f, Arguments... args) {
    //    int threadsNum;
        #pragma omp teams distribute parallel for num_threads(config.gridSize * config.blockSize)
        for (int i = 0; i < size; i++) {
            f(i, args...);
     //       threadsNum = omp_get_num_threads();
        }
    //    printf("Using %d OpenMP threads \n",threadsNum);
    }

    template<typename Function, typename... Arguments>
    void offload_map(int size, Function f, Arguments... args) {
        //  printf("Num Threads in map: %d\n", c.gridSize * c.blockSize);
        offload_map(vecpar::omp::getDefaultConfig(size), size, f, args...);
    }

    /// based on article:
    /// https://coderwall.com/p/gocbhg/openmp-improve-reduction-techniques
    template<typename R, typename Function>
    void offload_reduce(int size, R* result, Function f, vecmem::vector<R>& map_result) {
        #pragma omp parallel
        {
            R* tmp_result = new R();
            #pragma omp for nowait
            for (int i = 0; i < size; i++)
                f(tmp_result, map_result[i]);

            #pragma omp critical
                f(result, *tmp_result);
        }
    }

    template<typename R, typename Function, typename... Arguments>
    void offload_filter(int size, R* result, Function f, Arguments... args) {
        int idx = 0;
        #pragma omp teams distribute parallel for
        for (int i = 0; i < size; i++) {
            #pragma omp critical
                f(i, idx, *result, args...);
        }
        result->resize(idx);

    }
}
#endif //VECPAR_OMP_INTERNAL_HPP
