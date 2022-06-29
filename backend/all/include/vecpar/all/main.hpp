#ifndef VECPAR_MAIN_HPP
#define VECPAR_MAIN_HPP

#include <vecmem/containers/vector.hpp>

#include "vecpar/core/algorithms/parallelizable_filter.hpp"
#include "vecpar/core/algorithms/parallelizable_map.hpp"
#include "vecpar/core/algorithms/parallelizable_map_filter.hpp"
#include "vecpar/core/algorithms/parallelizable_map_reduce.hpp"
#include "vecpar/core/algorithms/parallelizable_reduce.hpp"
#include "vecpar/core/definitions/config.hpp"

#include "internal.hpp"

namespace vecpar {

template <class MemoryResource, class Algorithm,
          class R = typename Algorithm::intermediate_result_t, class T,
          typename... Arguments,
          typename std::enable_if_t<
              std::is_base_of<
                  vecpar::algorithm::parallelizable_map_1<R, T, Arguments...>,
                  Algorithm>::value ||
                  std::is_base_of<
                      vecpar::algorithm::parallelizable_mmap_1<T, Arguments...>,
                      Algorithm>::value,
              bool> = true>
R &parallel_algorithm(Algorithm algorithm, MemoryResource &mr,
                      vecpar::config config, T &data, Arguments... args) {

  return vecpar::parallel_map(algorithm, mr, config, data, args...);
}

template <class MemoryResource, class Algorithm,
          class R = typename Algorithm::intermediate_result_t, class T,
          typename... Arguments,
          typename std::enable_if_t<
              std::is_base_of<
                  vecpar::algorithm::parallelizable_map_1<R, T, Arguments...>,
                  Algorithm>::value ||
                  std::is_base_of<
                      vecpar::algorithm::parallelizable_mmap_1<T, Arguments...>,
                      Algorithm>::value,
              bool> = true>
R &parallel_algorithm(Algorithm algorithm, MemoryResource &mr, T &data,
                      Arguments... args) {

  return vecpar::parallel_map(algorithm, mr, data, args...);
}

template <class MemoryResource, class Algorithm, class T, typename... Arguments,
          typename std::enable_if_t<
              std::is_base_of<vecpar::algorithm::parallelizable_filter<T>,
                              Algorithm>::value,
              bool> = true>
T &parallel_algorithm(Algorithm algorithm, MemoryResource &mr, T &data) {

  return vecpar::parallel_filter(algorithm, mr, data);
}

template <class MemoryResource, class Algorithm, class R, typename... Arguments,
          typename std::enable_if_t<
              std::is_base_of<vecpar::algorithm::parallelizable_reduce<R>,
                              Algorithm>::value,
              bool> = true>
typename R::value_type &parallel_algorithm(Algorithm algorithm,
                                           MemoryResource &mr, R &data) {

  return vecpar::parallel_reduce(algorithm, mr, data);
}

template <
    class MemoryResource, class Algorithm,
    class R = typename Algorithm::result_t, class T, typename... Arguments,
    typename std::enable_if_t<
        std::is_base_of<
            vecpar::algorithm::parallelizable_map_filter_1<R, T, Arguments...>,
            Algorithm>::value ||
            std::is_base_of<vecpar::algorithm::parallelizable_mmap_filter_1<
                                T, Arguments...>,
                            Algorithm>::value,
        bool> = true>
R &parallel_algorithm(Algorithm algorithm, MemoryResource &mr,
                      vecpar::config config, T &data, Arguments... args) {

  return vecpar::parallel_map_filter(algorithm, mr, config, data, args...);
}

template <
    class MemoryResource, class Algorithm,
    class R = typename Algorithm::result_t, class T, typename... Arguments,
    typename std::enable_if_t<
        std::is_base_of<
            vecpar::algorithm::parallelizable_map_filter_1<R, T, Arguments...>,
            Algorithm>::value ||
            std::is_base_of<vecpar::algorithm::parallelizable_mmap_filter_1<
                                T, Arguments...>,
                            Algorithm>::value,
        bool> = true>
R &parallel_algorithm(Algorithm algorithm, MemoryResource &mr, T &data,
                      Arguments... args) {

  return vecpar::parallel_map_filter(algorithm, mr, data, args...);
}

template <
    class MemoryResource, class Algorithm,
    typename R = typename Algorithm::intermediate_result_t,
    class Result = typename Algorithm::result_t, class T, typename... Arguments,
    typename std::enable_if_t<
        std::is_base_of<vecpar::algorithm::parallelizable_map_reduce_1<
                            Result, R, T, Arguments...>,
                        Algorithm>::value ||
            std::is_base_of<vecpar::algorithm::parallelizable_mmap_reduce_1<
                                Result, T, Arguments...>,
                            Algorithm>::value,
        bool> = true>
Result &parallel_algorithm(Algorithm algorithm, MemoryResource &mr,
                           vecpar::config config, T &data, Arguments... args) {

  return vecpar::parallel_map_reduce(algorithm, mr, config, data, args...);
}

template <
    class MemoryResource, class Algorithm,
    typename R = typename Algorithm::intermediate_result_t,
    class Result = typename Algorithm::result_t, class T, typename... Arguments,
    typename std::enable_if_t<
        std::is_base_of<vecpar::algorithm::parallelizable_map_reduce_1<
                            Result, R, T, Arguments...>,
                        Algorithm>::value ||
            std::is_base_of<vecpar::algorithm::parallelizable_mmap_reduce_1<
                                Result, T, Arguments...>,
                            Algorithm>::value,
        bool> = true>
Result &parallel_algorithm(Algorithm algorithm, MemoryResource &mr, T &data,
                           Arguments... args) {

  return vecpar::parallel_map_reduce(algorithm, mr, data, args...);
}
} // namespace vecpar

#endif // VECPAR_MAIN_HPP
