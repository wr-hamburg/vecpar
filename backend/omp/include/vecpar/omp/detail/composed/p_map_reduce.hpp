#ifndef VECPAR_P_MAP_REDUCE_HPP
#define VECPAR_P_MAP_REDUCE_HPP

#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/containers/vector.hpp>

#include "vecpar/core/definitions/config.hpp"
#include "vecpar/core/algorithms/parallelizable_map_reduce.hpp"

#include "vecpar/omp/detail/internal.hpp"
#include "vecpar/omp/detail/simple/p_map.hpp"
#include "vecpar/omp/detail/simple/p_reduce.hpp"

namespace vecpar::omp {

    /// 1 iterable collection and config
    template <class Algorithm, typename R, typename T, typename... Arguments>
    R &parallel_map_reduce(Algorithm &algorithm, vecmem::memory_resource &mr,
                           vecmem::vector<T> data, Arguments... args) {

        return vecpar::omp::parallel_reduce(algorithm, mr,
                                            vecpar::omp::parallel_map(algorithm, mr, data, args...));
    }

    /// 1 iterable collection and no/default config
    template <class Algorithm, typename R, typename T, typename... Arguments>
    R &parallel_map_reduce(Algorithm &algorithm, vecmem::memory_resource &mr,
                           vecpar::config config, vecmem::vector<T> data,
                           Arguments... args) {

        return vecpar::omp::parallel_reduce(algorithm, mr,
                                            vecpar::omp::parallel_map(algorithm, mr, config, data, args...));
    }

    /// 2 iterable collections and config
    template <class Algorithm, typename R, typename T1, typename T2, typename... Arguments>
    R &parallel_map_reduce(Algorithm &algorithm, vecmem::memory_resource &mr,
                           vecmem::vector<T1> inout_1, vecmem::vector<T2> in_2, Arguments... args) {

        return vecpar::omp::parallel_reduce(algorithm, mr,
                                            vecpar::omp::parallel_map(algorithm, mr, inout_1, in_2, args...));
    }

    /// 2 iterable collections and no/default config
    template <class Algorithm, typename R, typename T1, typename T2, typename... Arguments>
    R &parallel_map_reduce(Algorithm &algorithm, vecmem::memory_resource &mr,
                           vecpar::config config, vecmem::vector<T1> inout_1,
                           vecmem::vector<T2> in_2,
                           Arguments... args) {

        return vecpar::omp::parallel_reduce(algorithm, mr,
                                            vecpar::omp::parallel_map(algorithm, mr, config, inout_1, in_2, args...));
    }

    /// 3 iterable collections and config
    template <class Algorithm, typename R, typename T1, typename T2, typename T3, typename... Arguments>
    R &parallel_map_reduce(Algorithm &algorithm, vecmem::memory_resource &mr,
                           vecmem::vector<T1> inout_1,
                           vecmem::vector<T2> in_2,
                           vecmem::vector<T3> in_3,
                           Arguments... args) {

        return vecpar::omp::parallel_reduce(algorithm, mr,
                                            vecpar::omp::parallel_map(algorithm, mr, inout_1, in_2, in_3, args...));
    }

    /// 3 iterable collections and no/default config
    template <class Algorithm, typename R, typename T1, typename T2, typename T3, typename... Arguments>
    R &parallel_map_reduce(Algorithm &algorithm, vecmem::memory_resource &mr,
                           vecpar::config config,
                           vecmem::vector<T1> inout_1,
                           vecmem::vector<T2> in_2,
                           vecmem::vector<T3> in_3,
                           Arguments... args) {

        return vecpar::omp::parallel_reduce(algorithm, mr,
                                            vecpar::omp::parallel_map(algorithm, mr, config, inout_1, in_2, in_3, args...));
    }

}
#endif //VECPAR_P_MAP_REDUCE_HPP
