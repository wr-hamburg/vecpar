#ifndef VECPAR_OMP_PARALLELIZATION_HPP
#define VECPAR_OMP_PARALLELIZATION_HPP

#include <cmath>
#include <omp.h>
#include <type_traits>
#include <utility>

#include "vecpar/core/definitions/config.hpp"
#include "vecpar/core/algorithms/detail/map.hpp"
#include "vecpar/core/algorithms/parallelizable_map.hpp"
#include "vecpar/core/algorithms/parallelizable_filter.hpp"
#include "vecpar/core/algorithms/parallelizable_reduce.hpp"
#include "vecpar/core/algorithms/parallelizable_map_reduce.hpp"
#include "vecpar/core/algorithms/parallelizable_map_filter.hpp"

#include "detail/internal.hpp"

namespace vecpar::omp {

    template<class Algorithm,
            class R = typename Algorithm::result_type,
            typename T,
            typename... Arguments,
            typename std::enable_if<std::is_base_of<detail::parallel_mmap<R, Arguments...>, Algorithm>::value>::type* = nullptr>
    vecmem::vector<R>* parallel_map(Algorithm& algorithm,
                                    __attribute__((unused)) vecmem::memory_resource& mr,
                                    vecmem::vector<T>& data,
                                    Arguments... args) {
        internal::offload_map(data.size(),
                              [&] (int idx) mutable {
                                  algorithm.map(data[idx], args...);
                              });
        return new vecmem::vector<R>(data);
    }

    template<class Algorithm,
            class R = typename Algorithm::result_type,
            typename T,
            typename... Arguments,
            typename std::enable_if<std::is_base_of<detail::parallel_map<R, T, Arguments...>, Algorithm>::value>::type* = nullptr>
    vecmem::vector<R>* parallel_map(Algorithm& algorithm,
                                    vecmem::memory_resource& mr,
                                    vecmem::vector<T>& data,
                                    Arguments... args) {
        vecmem::vector<R>* map_result = new vecmem::vector<R>(data.size(), &mr);
        internal::offload_map(data.size(),
                              [&] (int idx) mutable {
                                      algorithm.map(map_result->at(idx), data[idx], args...);
                              });
        return map_result;
    }

    template<typename Algorithm, typename R>
    R* parallel_reduce(Algorithm algorithm,
                       __attribute__((unused)) vecmem::memory_resource& mr,
                       vecmem::vector<R>& data) {
        R* result = new R();
        internal::offload_reduce(data.size(), result,
                                 [&] (R* r, R tmp) mutable {
                                     algorithm.reduce(r, tmp);
                                 }, data);

        return result;
    }

    template<typename Algorithm, typename T>
    vecmem::vector<T>* parallel_filter(Algorithm algorithm,
                                       vecmem::memory_resource& mr,
                                       vecmem::vector<T>& data) {

        vecmem::vector<T> *result = new vecmem::vector<T>(data.size(), &mr);
        internal::offload_filter(data.size(), result,
                                 [&](int idx, int &result_index,
                                     vecmem::vector<T> &local_result) mutable {
                                   if (algorithm.filter(data[idx])) {
                                     local_result[result_index] = data[idx];
                                     result_index++;
                                   }
                                 });
        return result;
    }

    template <typename Function, typename... Arguments>
    void parallel_map(size_t size, Function f, Arguments... args) {
      internal::offload_map(size, f, args...);
    }

    template <typename Function, typename... Arguments>
    void parallel_reduce(size_t size, Function f, Arguments... args) {
      internal::offload_reduce(size, f, args...);
    }

    template<class Algorithm, typename R, typename T, typename... Arguments>
    R* parallel_map_reduce(Algorithm& algorithm,
                           vecmem::memory_resource& mr,
                           vecmem::vector<T> data,
                           Arguments... args)  {
        vecmem::vector<R>* map_result = parallel_map(algorithm, mr, data, args...);
        R* result = parallel_reduce(algorithm, mr, *map_result);
        return result;
    }

    template<class Algorithm, typename R, typename T, typename... Arguments>
    vecmem::vector<R>* parallel_map_filter(Algorithm& algorithm,
                                          vecmem::memory_resource& mr,
                                          vecmem::vector<T> data,
                                          Arguments... args)  {

        vecmem::vector<R>* map_result = parallel_map(algorithm, mr, data, args...);
        vecmem::vector<R>* result = parallel_filter(algorithm, mr, *map_result);
        return result;
    }

    template<class MemoryResource,
            class Algorithm,
            class R = typename Algorithm::result_type,
            class T, typename... Arguments,
            typename std::enable_if<std::is_base_of<vecpar::algorithm::parallelizable_map_filter<R, T, Arguments...>, Algorithm>::value>::type* = nullptr>
    vecmem::vector<R>* parallel_algorithm(Algorithm algorithm,
                                          MemoryResource& mr,
                                          vecmem::vector<T>& data,
                                          Arguments... args) {

        return parallel_map_filter<Algorithm, R, T, Arguments...>(algorithm, mr, data, args...);
    }

    template<class MemoryResource,
            class Algorithm,
            class R = typename Algorithm::result_type,
            class T, typename... Arguments,
            typename std::enable_if<std::is_base_of<vecpar::algorithm::parallelizable_map_reduce<R, T, Arguments...>, Algorithm>::value>::type* = nullptr>
    R* parallel_algorithm(Algorithm algorithm,
                          MemoryResource& mr,
                          vecmem::vector<T>& data,
                          Arguments... args) {

        return parallel_map_reduce<Algorithm, R, T, Arguments...>(algorithm, mr, data, args...);
    }
}
#endif //VECPAR_OMP_PARALLELIZATION_HPP
