#ifndef VECPAR_OMP_PARALLELIZATION_HPP
#define VECPAR_OMP_PARALLELIZATION_HPP

#include <type_traits>
#include <vecmem/memory/memory_resource.hpp>

#include "vecpar/core/algorithms/parallelizable_map_filter.hpp"
#include "vecpar/core/algorithms/parallelizable_map_reduce.hpp"
#include "vecpar/core/definitions/config.hpp"

#include "vecpar/omp/detail/composed/p_map_filter.hpp"
#include "vecpar/omp/detail/composed/p_map_reduce.hpp"

namespace vecpar::omp {

template <
    class MemoryResource, class Algorithm,
    class R = typename Algorithm::result_t, class T, typename... Arguments,
    typename std::enable_if_t<
        std::is_base_of<
            vecpar::algorithm::parallelizable_map_filter<R, T, Arguments...>,
            Algorithm>::value ||
            std::is_base_of<
                vecpar::algorithm::parallelizable_mmap_filter<T, Arguments...>,
                Algorithm>::value,
        bool> = true>
vecmem::vector<R> &parallel_algorithm(Algorithm algorithm, MemoryResource &mr,
                                      vecpar::config config,
                                      vecmem::vector<T> &data,
                                      Arguments... args) {

  return vecpar::omp::parallel_map_filter<Algorithm, R, T, Arguments...>(
      algorithm, mr, config, data, args...);
}

template <
    class MemoryResource, class Algorithm,
    class R = typename Algorithm::result_t, class T, typename... Arguments,
    typename std::enable_if_t<
        std::is_base_of<
            vecpar::algorithm::parallelizable_map_filter<R, T, Arguments...>,
            Algorithm>::value ||
            std::is_base_of<
                vecpar::algorithm::parallelizable_mmap_filter<T, Arguments...>,
                Algorithm>::value,
        bool> = true>
vecmem::vector<R> &parallel_algorithm(Algorithm algorithm, MemoryResource &mr,
                                      vecmem::vector<T> &data,
                                      Arguments... args) {

  return vecpar::omp::parallel_map_filter<Algorithm, R, T, Arguments...>(
      algorithm, mr, omp::getDefaultConfig(), data, args...);
}

template <
    class MemoryResource, class Algorithm,
    class R = typename Algorithm::result_t, class T, typename... Arguments,
    typename std::enable_if_t<
        std::is_base_of<
            vecpar::algorithm::parallelizable_map_reduce<R, T, Arguments...>,
            Algorithm>::value ||
            std::is_base_of<
                vecpar::algorithm::parallelizable_mmap_reduce<T, Arguments...>,
                Algorithm>::value,
        bool> = true>
R &parallel_algorithm(Algorithm algorithm, MemoryResource &mr,
                      vecpar::config config, vecmem::vector<T> &data,
                      Arguments... args) {

  return vecpar::omp::parallel_map_reduce<Algorithm, R, T, Arguments...>(
      algorithm, mr, config, data, args...);
}

template <
    class MemoryResource, class Algorithm,
    class R = typename Algorithm::result_t, class T, typename... Arguments,
    typename std::enable_if_t<
        std::is_base_of<
            vecpar::algorithm::parallelizable_map_reduce<R, T, Arguments...>,
            Algorithm>::value ||
            std::is_base_of<
                vecpar::algorithm::parallelizable_mmap_reduce<T, Arguments...>,
                Algorithm>::value,
        bool> = true>
R &parallel_algorithm(Algorithm algorithm, MemoryResource &mr,
                      vecmem::vector<T> &data, Arguments... args) {

  return vecpar::omp::parallel_map_reduce<Algorithm, R, T, Arguments...>(
      algorithm, mr, omp::getDefaultConfig(), data, args...);
}

/// multiple collections
    template <
            class MemoryResource, class Algorithm,
            class R = typename Algorithm::result_t, class T1, class T2, typename... Arguments,
            typename std::enable_if_t<
                    std::is_base_of<
                            vecpar::algorithm::parallelizable_map_filter<R, T1, T2, Arguments...>,
                            Algorithm>::value ||
                    std::is_base_of<
                            vecpar::algorithm::parallelizable_mmap_filter<T1, T2, Arguments...>,
                            Algorithm>::value,
                    bool> = true>
    vecmem::vector<R> &parallel_algorithm(Algorithm algorithm, MemoryResource &mr,
                                          vecpar::config config,
                                          vecmem::vector<T1> &inout_1,
                                          vecmem::vector<T2> &in_2,
                                          Arguments... args) {

        return vecpar::omp::parallel_map_filter<Algorithm, R, T1, T2, Arguments...>(
                algorithm, mr, config, inout_1, in_2, args...);
    }

    template <
            class MemoryResource, class Algorithm,
            class R = typename Algorithm::result_t, class T1, class T2, typename... Arguments,
            typename std::enable_if_t<
                    std::is_base_of<
                            vecpar::algorithm::parallelizable_map_filter<R, T1, T2, Arguments...>,
                            Algorithm>::value ||
                    std::is_base_of<
                            vecpar::algorithm::parallelizable_mmap_filter<T1, T2, Arguments...>,
                            Algorithm>::value,
                    bool> = true>
    vecmem::vector<R> &parallel_algorithm(Algorithm algorithm, MemoryResource &mr,
                                          vecmem::vector<T1> &inout_1,
                                          vecmem::vector<T2> &in_2,
                                          Arguments... args) {

        return vecpar::omp::parallel_map_filter<Algorithm, R, T1, T2, Arguments...>(
                algorithm, mr, omp::getDefaultConfig(), inout_1, in_2, args...);
    }

    template <
            class MemoryResource, class Algorithm,
            class R = typename Algorithm::result_t, class T1, class T2, typename... Arguments,
            typename std::enable_if_t<
                    std::is_base_of<
                            vecpar::algorithm::parallelizable_map_reduce<R, T1, T2, Arguments...>,
                            Algorithm>::value ||
                    std::is_base_of<
                            vecpar::algorithm::parallelizable_mmap_reduce<T1, T2, Arguments...>,
                            Algorithm>::value,
                    bool> = true>
    R &parallel_algorithm(Algorithm algorithm, MemoryResource &mr,
                                          vecpar::config config,
                                          vecmem::vector<T1> &inout_1,
                                          vecmem::vector<T2> &in_2,
                                          Arguments... args) {

        return vecpar::omp::parallel_map_reduce<Algorithm, R, T1, T2, Arguments...>(
                algorithm, mr, config, inout_1, in_2, args...);
    }

    template <
            class MemoryResource, class Algorithm,
            class R = typename Algorithm::result_t, class T1, class T2, typename... Arguments,
            typename std::enable_if_t<
                    std::is_base_of<
                            vecpar::algorithm::parallelizable_map_reduce<R, T1, T2, Arguments...>,
                            Algorithm>::value ||
                    std::is_base_of<
                            vecpar::algorithm::parallelizable_mmap_reduce<T1, T2, Arguments...>,
                            Algorithm>::value,
                    bool> = true>
    R &parallel_algorithm(Algorithm algorithm, MemoryResource &mr,
                                          vecmem::vector<T1> &inout_1,
                                          vecmem::vector<T2> &in_2,
                                          Arguments... args) {

        return vecpar::omp::parallel_map_reduce<Algorithm, R, T1, T2, Arguments...>(
                algorithm, mr, omp::getDefaultConfig(), inout_1, in_2, args...);
    }

    template <
            class MemoryResource, class Algorithm,
            class R = typename Algorithm::result_t, class T1, class T2, class T3, typename... Arguments,
            typename std::enable_if_t<
                    std::is_base_of<
                            vecpar::algorithm::parallelizable_map_filter<R, T1, T2, T3, Arguments...>,
                            Algorithm>::value ||
                    std::is_base_of<
                            vecpar::algorithm::parallelizable_mmap_filter<T1, T2, T3, Arguments...>,
                            Algorithm>::value,
                    bool> = true>
    vecmem::vector<R> &parallel_algorithm(Algorithm algorithm, MemoryResource &mr,
                                          vecpar::config config,
                                          vecmem::vector<T1> &inout_1,
                                          vecmem::vector<T2> &in_2,
                                          vecmem::vector<T3> &in_3,
                                          Arguments... args) {

        return vecpar::omp::parallel_map_filter<Algorithm, R, T1, T2, T3, Arguments...>(
                algorithm, mr, config, inout_1, in_2, in_3, args...);
    }

    template <
            class MemoryResource, class Algorithm,
            class R = typename Algorithm::result_t, class T1, class T2, class T3, typename... Arguments,
            typename std::enable_if_t<
                    std::is_base_of<
                            vecpar::algorithm::parallelizable_map_filter<R, T1, T2, T3, Arguments...>,
                            Algorithm>::value ||
                    std::is_base_of<
                            vecpar::algorithm::parallelizable_mmap_filter<T1, T2, T3, Arguments...>,
                            Algorithm>::value,
                    bool> = true>
    vecmem::vector<R> &parallel_algorithm(Algorithm algorithm, MemoryResource &mr,
                                          vecmem::vector<T1> &inout_1,
                                          vecmem::vector<T2> &in_2,
                                          vecmem::vector<T3> &in_3,
                                          Arguments... args) {

        return vecpar::omp::parallel_map_filter<Algorithm, R, T1, T2, T3, Arguments...>(
                algorithm, mr, omp::getDefaultConfig(), inout_1, in_2, in_3, args...);
    }

    template <
            class MemoryResource, class Algorithm,
            class R = typename Algorithm::result_t, class T1, class T2, class T3, typename... Arguments,
            typename std::enable_if_t<
                    std::is_base_of<
                            vecpar::algorithm::parallelizable_map_reduce<R, T1, T2, T3, Arguments...>,
                            Algorithm>::value ||
                    std::is_base_of<
                            vecpar::algorithm::parallelizable_mmap_reduce<T1, T2, T3, Arguments...>,
                            Algorithm>::value,
                    bool> = true>
    R &parallel_algorithm(Algorithm algorithm, MemoryResource &mr,
                                          vecpar::config config,
                                          vecmem::vector<T1> &inout_1,
                                          vecmem::vector<T2> &in_2,
                                          vecmem::vector<T3> &in_3,
                                          Arguments... args) {

        return vecpar::omp::parallel_map_reduce<Algorithm, R, T1, T2, T3, Arguments...>(
                algorithm, mr, config, inout_1, in_2, in_3, args...);
    }

    template <
            class MemoryResource, class Algorithm,
            class R = typename Algorithm::result_t, typename T1, typename T2, typename T3, typename... Arguments,
            typename std::enable_if_t<
                    std::is_base_of<
                            vecpar::algorithm::parallelizable_map_reduce<R, T1, T2, T3, Arguments...>,
                            Algorithm>::value ||
                    std::is_base_of<
                            vecpar::algorithm::parallelizable_mmap_reduce<T1, T2, T3, Arguments...>,
                            Algorithm>::value,
                    bool> = true>
    R &parallel_algorithm(Algorithm algorithm, MemoryResource &mr,
                                          vecmem::vector<T1> &inout_1,
                                          vecmem::vector<T2> in_2,
                                          vecmem::vector<T3> in_3,
                                          Arguments... args) {

        return vecpar::omp::parallel_map_reduce<Algorithm, R, T1, T2, T3, Arguments...>(
                algorithm, mr, omp::getDefaultConfig(), inout_1, in_2, in_3, args...);
    }
} // namespace vecpar::omp
#endif // VECPAR_OMP_PARALLELIZATION_HPP
