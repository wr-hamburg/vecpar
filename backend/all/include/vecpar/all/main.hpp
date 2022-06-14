#ifndef VECPAR_MAIN_HPP
#define VECPAR_MAIN_HPP

#include <vecmem/containers/vector.hpp>

#include "vecpar/core/algorithms/parallelizable_map_filter.hpp"
#include "vecpar/core/algorithms/parallelizable_map_reduce.hpp"
#include "vecpar/core/definitions/config.hpp"

#include "internal.hpp"

namespace vecpar {

     template<class MemoryResource,
            class Algorithm,
            class R = typename Algorithm::result_t,
            class T, typename... Arguments,
            typename std::enable_if_t<std::is_base_of<vecpar::algorithm::parallelizable_map<R, T, Arguments...>, Algorithm>::value ||
                                      std::is_base_of<vecpar::algorithm::parallelizable_mmap<T, Arguments...>, Algorithm>::value, bool> = true>
    vecmem::vector<R>& parallel_algorithm(Algorithm algorithm,
                                          MemoryResource& mr,
                                          vecpar::config config,
                                          vecmem::vector<T>& data,
                                          Arguments... args) {

        return vecpar::parallel_map(algorithm, mr, config, data, args...);
    }

    template<class MemoryResource,
            class Algorithm,
            class R = typename Algorithm::result_t,
            class T, typename... Arguments,
            typename std::enable_if_t<std::is_base_of<vecpar::algorithm::parallelizable_map<R, T, Arguments...>, Algorithm>::value ||
                                      std::is_base_of<vecpar::algorithm::parallelizable_mmap<T, Arguments...>, Algorithm>::value, bool> = true>
    vecmem::vector<R>& parallel_algorithm(Algorithm algorithm,
                                          MemoryResource& mr,
                                          vecmem::vector<T>& data,
                                          Arguments... args) {

        return vecpar::parallel_map(algorithm, mr, data, args...);
    }

    template<class MemoryResource,
            class Algorithm,
            class T, typename... Arguments,
            typename std::enable_if_t<std::is_base_of<vecpar::algorithm::parallelizable_filter<T>, Algorithm>::value, bool> = true>
        vecmem::vector<T>& parallel_algorithm(Algorithm algorithm,
                                          MemoryResource& mr,
                                          vecmem::vector<T>& data) {

        return vecpar::parallel_filter(algorithm, mr, data);
    }

    template<class MemoryResource,
            class Algorithm,
            class R, typename... Arguments,
            typename std::enable_if_t<std::is_base_of<vecpar::algorithm::parallelizable_reduce<R>, Algorithm>::value, bool> = true>
        R& parallel_algorithm(Algorithm algorithm,
                                              MemoryResource& mr,
                                              vecmem::vector<R>& data) {

        return vecpar::parallel_reduce(algorithm, mr, data);
    }

    template<class MemoryResource,
            class Algorithm,
            class R = typename Algorithm::result_t,
            class T, typename... Arguments,
            typename std::enable_if_t<std::is_base_of<vecpar::algorithm::parallelizable_map_filter<R, T, Arguments...>, Algorithm>::value ||
                                      std::is_base_of<vecpar::algorithm::parallelizable_mmap_filter<T, Arguments...>, Algorithm>::value, bool> = true>
    vecmem::vector<R>& parallel_algorithm(Algorithm algorithm,
                                         MemoryResource& mr,
                                         vecpar::config config,
                                         vecmem::vector<T>& data,
                                         Arguments... args) {

        return vecpar::parallel_map_filter(algorithm, mr, config, data, args...);
    }

    template<class MemoryResource,
            class Algorithm,
            class R = typename Algorithm::result_t,
            class T, typename... Arguments,
            typename std::enable_if_t<std::is_base_of<vecpar::algorithm::parallelizable_map_filter<R, T, Arguments...>, Algorithm>::value ||
                                      std::is_base_of<vecpar::algorithm::parallelizable_mmap_filter<T, Arguments...>, Algorithm>::value, bool> = true>
    vecmem::vector<R>& parallel_algorithm(Algorithm algorithm,
                                          MemoryResource& mr,
                                          vecmem::vector<T>& data,
                                          Arguments... args) {

        return vecpar::parallel_map_filter(algorithm, mr, data, args...);
    }

    template<class MemoryResource,
            class Algorithm,
            class R = typename Algorithm::result_t,
            class T, typename... Arguments,
            typename std::enable_if_t<std::is_base_of<vecpar::algorithm::parallelizable_map_reduce<R, T, Arguments...>, Algorithm>::value ||
                                      std::is_base_of<vecpar::algorithm::parallelizable_mmap_reduce<T, Arguments...>, Algorithm>::value, bool> = true>
    R& parallel_algorithm(Algorithm algorithm,
                         MemoryResource& mr,
                         vecpar::config config,
                         vecmem::vector<T>& data,
                         Arguments... args) {

        return vecpar::parallel_map_reduce(algorithm, mr, config, data, args...);
    }

    template<class MemoryResource,
            class Algorithm,
            class R = typename Algorithm::result_t,
            class T, typename... Arguments,
            typename std::enable_if_t<std::is_base_of<vecpar::algorithm::parallelizable_map_reduce<R, T, Arguments...>, Algorithm>::value ||
                                      std::is_base_of<vecpar::algorithm::parallelizable_mmap_reduce<T, Arguments...>, Algorithm>::value, bool> = true>
    R& parallel_algorithm(Algorithm algorithm,
                          MemoryResource& mr,
                          vecmem::vector<T>& data,
                          Arguments... args) {

        return vecpar::parallel_map_reduce(algorithm, mr, data, args...);
    }
}

#endif //VECPAR_MAIN_HPP
