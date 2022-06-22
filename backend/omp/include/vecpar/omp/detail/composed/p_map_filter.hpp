#ifndef VECPAR_P_MAP_FILTER_HPP
#define VECPAR_P_MAP_FILTER_HPP

#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/containers/vector.hpp>

#include "vecpar/core/definitions/config.hpp"
#include "vecpar/core/algorithms/parallelizable_map_filter.hpp"

#include "vecpar/omp/detail/internal.hpp"
#include "vecpar/omp/detail/simple/p_map.hpp"
#include "vecpar/omp/detail/simple/p_filter.hpp"

namespace vecpar::omp {

    /// 1 iterable collection & config
    template <class Algorithm, typename R, typename T, typename... Arguments>
    vecmem::vector<R> &
    parallel_map_filter(Algorithm &algorithm, vecmem::memory_resource &mr,
                        vecpar::config config, vecmem::vector<T> data,
                        Arguments... args) {

        return vecpar::omp::parallel_filter(algorithm, mr,
                                            vecpar::omp::parallel_map(algorithm, mr, config, data, args...));
    }

    /// 1 iterable collection and no/default config
    template <class Algorithm, typename R, typename T, typename... Arguments>
    vecmem::vector<R> &
    parallel_map_filter(Algorithm &algorithm, vecmem::memory_resource &mr,
                        vecmem::vector<T> data, Arguments... args) {

        return vecpar::omp::parallel_map_filter<Algorithm, R, T, Arguments...>(
                algorithm, mr, omp::getDefaultConfig(), data, args...);
    }

    /// 2 iterable collections and config
    template <class Algorithm, typename R, typename T1, typename T2, typename... Arguments>
    vecmem::vector<R> &
    parallel_map_filter(Algorithm &algorithm, vecmem::memory_resource &mr,
                        vecpar::config config, vecmem::vector<T1> inout_1,
                        vecmem::vector<T2> in_2,
                        Arguments... args) {

        return vecpar::omp::parallel_filter(algorithm, mr,
                                            vecpar::omp::parallel_map(algorithm, mr, config, inout_1, in_2, args...));
    }

    /// 2 iterable collections and no/default config
    template <class Algorithm, typename R, typename T1, typename T2, typename... Arguments>
    vecmem::vector<R> &
    parallel_map_filter(Algorithm &algorithm, vecmem::memory_resource &mr,
                        vecmem::vector<T1> inout_1,
                        vecmem::vector<T2> in_2,
                        Arguments... args) {

        return vecpar::omp::parallel_map_filter<Algorithm, R, T1, T2, Arguments...>(
                algorithm, mr, omp::getDefaultConfig(), inout_1, in_2, args...);
    }

    /// 3 iterable collections and config
    template <class Algorithm, typename R, typename T1, typename T2, typename T3, typename... Arguments>
    vecmem::vector<R> &
    parallel_map_filter(Algorithm &algorithm, vecmem::memory_resource &mr,
                        vecpar::config config, vecmem::vector<T1> inout_1,
                        vecmem::vector<T2> in_2,
                        vecmem::vector<T3> in_3,
                        Arguments... args) {

        return vecpar::omp::parallel_filter(algorithm, mr,
                                            vecpar::omp::parallel_map(algorithm, mr, config, inout_1, in_2, in_3, args...));
    }

    /// 3 iterable collections and no/default config
    template <class Algorithm, typename R, typename T1, typename T2, typename T3, typename... Arguments>
    vecmem::vector<R> &
    parallel_map_filter(Algorithm &algorithm, vecmem::memory_resource &mr,
                        vecmem::vector<T1> inout_1,
                        vecmem::vector<T2> in_2,
                        vecmem::vector<T3> in_3,
                        Arguments... args) {

        return vecpar::omp::parallel_map_filter<Algorithm, R, T1, T2, T3, Arguments...>(
                algorithm, mr, omp::getDefaultConfig(), inout_1, in_2, in_3, args...);
    }
}

#endif //VECPAR_P_MAP_FILTER_HPP
