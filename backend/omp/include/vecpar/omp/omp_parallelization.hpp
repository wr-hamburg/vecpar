#ifndef VECPAR_OMP_PARALLELIZATION_HPP
#define VECPAR_OMP_PARALLELIZATION_HPP

#include <cmath>
#include <omp.h>
#include <type_traits>
#include <utility>

#include <vecmem/memory/memory_resource.hpp>

#include "vecpar/core/algorithms/parallelizable_map.hpp"
#include "vecpar/core/algorithms/parallelizable_map_filter.hpp"
#include "vecpar/core/algorithms/parallelizable_map_reduce.hpp"
#include "vecpar/core/definitions/config.hpp"

#include "vecpar/omp/detail/internal.hpp"

namespace vecpar::omp {

/// default offloading generic functions
template <typename Function, typename... Arguments>
void parallel_map(vecpar::config config, size_t size, Function f,
                  Arguments... args) {
  internal::offload_map(config, size, f, args...);
}

template <typename Function, typename... Arguments>
void parallel_map(size_t size, Function f, Arguments... args) {
  parallel_map(omp::getDefaultConfig(), size, f, args...);
}

template <typename Function, typename... Arguments>
void parallel_reduce(size_t size, Function f, Arguments... args) {
  internal::offload_reduce(size, f, args...);
}

/// specific simple implementations
template <class Algorithm,
          typename R = typename Algorithm::intermediate_result_t, typename T,
          typename... Arguments,
          typename std::enable_if<
              std::is_base_of<vecpar::detail::parallel_mmap_1<R, Arguments...>,
                              Algorithm>::value>::type * = nullptr>
R &parallel_map(Algorithm &algorithm,
                __attribute__((unused)) vecmem::memory_resource &mr,
                vecpar::config config, T &data, Arguments... args) {
  internal::offload_map(config, data.size(), [&](int idx) mutable {
    algorithm.map(data[idx], args...);
  });
  return data;
}

template <class Algorithm,
          typename R = typename Algorithm::intermediate_result_t, typename T,
          typename... Arguments,
          typename std::enable_if<
              std::is_base_of<vecpar::detail::parallel_mmap_1<R, Arguments...>,
                              Algorithm>::value>::type * = nullptr>
R &parallel_map(Algorithm &algorithm,
                __attribute__((unused)) vecmem::memory_resource &mr, T &data,
                Arguments... args) {
  return vecpar::omp::parallel_map(algorithm, mr, omp::getDefaultConfig(), data,
                                   args...);
}

template <
    class Algorithm, typename R = typename Algorithm::intermediate_result_t,
    typename T, typename... Arguments,
    typename std::enable_if<
        std::is_base_of<vecpar::detail::parallel_map_1<R, T, Arguments...>,
                        Algorithm>::value>::type * = nullptr>
R &parallel_map(Algorithm &algorithm,
                __attribute__((unused)) vecmem::memory_resource &mr,
                vecpar::config config, T &data, Arguments... args) {
  R *map_result = new R(data.size(), &mr);
  internal::offload_map(config, data.size(), [&](int idx) mutable {
    algorithm.map(map_result->at(idx), data[idx], args...);
  });
  return *map_result;
}

template <
    class Algorithm, typename R = typename Algorithm::intermediate_result_t,
    typename T, typename... Arguments,
    typename std::enable_if<
        std::is_base_of<vecpar::detail::parallel_map_1<R, T, Arguments...>,
                        Algorithm>::value>::type * = nullptr>
R &parallel_map(Algorithm &algorithm,
                __attribute__((unused)) vecmem::memory_resource &mr, T &data,
                Arguments... args) {
  return vecpar::omp::parallel_map(algorithm, mr, omp::getDefaultConfig(), data,
                                   args...);
}

template <typename Algorithm, typename R>
typename R::value_type &
parallel_reduce(Algorithm algorithm,
                __attribute__((unused)) vecmem::memory_resource &mr, R &data) {
  typename R::value_type *result = new typename R::value_type();
  internal::offload_reduce(
      data.size(), result,
      [&](typename R::value_type *r, typename R::value_type tmp) mutable {
        algorithm.reduce(r, tmp);
      },
      data);

  return *result;
}

template <typename Algorithm, typename T>
T &parallel_filter(Algorithm algorithm, vecmem::memory_resource &mr, T &data) {

  T *result = new T(data.size(), &mr);
  internal::offload_filter(
      data.size(), result,
      [&](int idx, int &result_index, T &local_result) mutable {
        if (algorithm.filter(data[idx])) {
          local_result[result_index] = data[idx];
          result_index++;
        }
      });
  return *result;
}

/// specific composed implementations
template <class Algorithm, typename Result, typename R, typename T,
          typename... Arguments>
Result &parallel_map_reduce(Algorithm &algorithm, vecmem::memory_resource &mr,
                            T &data, Arguments... args) {

  return vecpar::omp::parallel_reduce(
      algorithm, mr, vecpar::omp::parallel_map(algorithm, mr, data, args...));
}

template <class Algorithm, typename Result, typename R, typename T,
          typename... Arguments>
Result &parallel_map_reduce(Algorithm &algorithm, vecmem::memory_resource &mr,
                            vecpar::config config, T &data, Arguments... args) {

  return vecpar::omp::parallel_reduce(
      algorithm, mr,
      vecpar::omp::parallel_map(algorithm, mr, config, data, args...));
}

template <class Algorithm, typename R, typename T, typename... Arguments>
R &parallel_map_filter(Algorithm &algorithm, vecmem::memory_resource &mr,
                       vecpar::config config, T &data, Arguments... args) {

  return vecpar::omp::parallel_filter<Algorithm, R>(
      algorithm, mr,
      vecpar::omp::parallel_map<Algorithm, R, T, Arguments...>(
          algorithm, mr, config, data, args...));
}

template <class Algorithm, typename R, typename T, typename... Arguments>
R &parallel_map_filter(Algorithm &algorithm, vecmem::memory_resource &mr,
                       T &data, Arguments... args) {

  return vecpar::omp::parallel_map_filter<Algorithm, R, T, Arguments...>(
      algorithm, mr, omp::getDefaultConfig(), data, args...);
}

template <
    class MemoryResource, class Algorithm,
    typename Result = typename Algorithm::result_t,
    typename R = typename Algorithm::intermediate_result_t, typename T,
    typename... Arguments,
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

  return vecpar::omp::parallel_map_reduce<Algorithm, Result, R, T,
                                          Arguments...>(algorithm, mr, config,
                                                        data, args...);
}

template <
    class MemoryResource, class Algorithm,
    typename R = typename Algorithm::intermediate_result_t,
    class Result = typename Algorithm::result_t, typename T,
    typename... Arguments,
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

  return vecpar::omp::parallel_map_reduce<Algorithm, Result, R, T,
                                          Arguments...>(
      algorithm, mr, omp::getDefaultConfig(), data, args...);
}

template <
    class MemoryResource, class Algorithm,
    class R = typename Algorithm::result_t, typename T, typename... Arguments,
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

  return vecpar::omp::parallel_map_filter<Algorithm, R, T, Arguments...>(
      algorithm, mr, config, data, args...);
}

template <
    class MemoryResource, class Algorithm,
    class R = typename Algorithm::result_t, typename T, typename... Arguments,
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

  return vecpar::omp::parallel_map_filter<Algorithm, R, T, Arguments...>(
      algorithm, mr, omp::getDefaultConfig(), data, args...);
}
} // namespace vecpar::omp
#endif // VECPAR_OMP_PARALLELIZATION_HPP
