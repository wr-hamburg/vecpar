#ifndef VECPAR_P_MAP_HPP
#define VECPAR_P_MAP_HPP

#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/containers/vector.hpp>

#include "vecpar/core/algorithms/parallelizable_map.hpp"
#include "vecpar/core/definitions/config.hpp"

#include "vecpar/omp/detail/internal.hpp"

namespace vecpar::omp {

    /// generic functions
    template <typename Function, typename... Arguments>
    void parallel_map(vecpar::config config, size_t size, Function f,
                      Arguments... args) {
        internal::offload_map(config, size, f, args...);
    }

    template <typename Function, typename... Arguments>
    void parallel_map(size_t size, Function f, Arguments... args) {
        parallel_map(omp::getDefaultConfig(), size, f, args...);
    }

    /// functions based on vecpar algorithms

    /// 1 iterable collection and config
    template <class Algorithm, class R = typename Algorithm::result_t, typename T,
            typename... Arguments,
            typename std::enable_if<std::is_base_of<
                    detail::parallel_mmap<R, Arguments...>, Algorithm>::value>::type
            * = nullptr>
    vecmem::vector<R> &parallel_map(Algorithm &algorithm,
                                    __attribute__((unused))
                                    vecmem::memory_resource &mr,
                                    vecpar::config config, vecmem::vector<T> &data,
                                    Arguments... args) {
        internal::offload_map(config, data.size(), [&](int idx) mutable {
            algorithm.map(data[idx], args...);
        });
        return data;
    }

    /// 1 iterable collections and no/default config
    template <class Algorithm, class R = typename Algorithm::result_t, typename T,
            typename... Arguments,
            typename std::enable_if<std::is_base_of<
                    detail::parallel_mmap<R, Arguments...>, Algorithm>::value>::type
            * = nullptr>
    vecmem::vector<R> &parallel_map(Algorithm &algorithm,
                                    __attribute__((unused))
                                    vecmem::memory_resource &mr,
                                    vecmem::vector<T> &data, Arguments... args) {
        return vecpar::omp::parallel_map(algorithm, mr, omp::getDefaultConfig(), data, args...);
    }

    /// 1 iterable collection and config
    template <class Algorithm, class R = typename Algorithm::result_t, typename T,
            typename... Arguments,
            typename std::enable_if<std::is_base_of<
                    detail::parallel_map<R, T, Arguments...>, Algorithm>::value>::type
            * = nullptr>
    vecmem::vector<R> &parallel_map(Algorithm &algorithm,
                                    vecmem::memory_resource &mr,
                                    vecpar::config config, vecmem::vector<T> &data,
                                    Arguments... args) {
        vecmem::vector<R> *map_result = new vecmem::vector<R>(data.size(), &mr);
        internal::offload_map(config, data.size(), [&](int idx) mutable {
            algorithm.map(map_result->at(idx), data[idx], args...);
        });
        return *map_result;
    }

    /// 1 iterable collection and no/default config
    template <class Algorithm, class R = typename Algorithm::result_t, typename T,
            typename... Arguments,
            typename std::enable_if<std::is_base_of<
                    detail::parallel_map<R, T, Arguments...>, Algorithm>::value>::type
            * = nullptr>
    vecmem::vector<R> &parallel_map(Algorithm &algorithm,
                                    vecmem::memory_resource &mr,
                                    vecmem::vector<T> &data, Arguments... args) {

        return vecpar::omp::parallel_map(algorithm, mr, omp::getDefaultConfig(), data, args...);
    }

    /// 2 iterable collections and config
    template <class Algorithm, class R = typename Algorithm::result_t, typename T1,
            typename T2,
            typename... Arguments,
            typename std::enable_if<std::is_base_of<
                    detail::parallel_mmap<T1, T2, Arguments...>, Algorithm>::value>::type
            * = nullptr>
    vecmem::vector<R> &parallel_map(Algorithm &algorithm,
                                    __attribute__((unused))
                                    vecmem::memory_resource &mr,
                                    vecpar::config config, vecmem::vector<T1> &mdata,
                                    vecmem::vector<T2> &data,
                                    Arguments... args) {
        internal::offload_map(config, data.size(), [&](int idx) mutable {
            algorithm.map(mdata[idx], data[idx], args...);
        });
        return mdata;
    }

    /// 2 iterable collections and no/default config
    template <class Algorithm, class R = typename Algorithm::result_t, typename T1,
            typename T2,
            typename... Arguments,
            typename std::enable_if<std::is_base_of<
                    detail::parallel_mmap<T1, T2, Arguments...>, Algorithm>::value>::type
            * = nullptr>
    vecmem::vector<R> &parallel_map(Algorithm &algorithm,
                                    __attribute__((unused))
                                    vecmem::memory_resource &mr,
                                    vecmem::vector<T1> &mdata,
                                    vecmem::vector<T2> &data,
                                    Arguments... args) {
        return vecpar::omp::parallel_map(algorithm, mr, omp::getDefaultConfig(), mdata, data, args...);
    }

    /// 3 iterable collections and config
    template <class Algorithm, class R = typename Algorithm::result_t, typename T1,
            typename T2, typename T3,
            typename... Arguments,
            typename std::enable_if<std::is_base_of<
                    detail::parallel_mmap<T1, T2, T3, Arguments...>, Algorithm>::value>::type
            * = nullptr>
    vecmem::vector<R> &parallel_map(Algorithm &algorithm,
                                    __attribute__((unused))
                                    vecmem::memory_resource &mr,
                                    vecpar::config config, vecmem::vector<T1> &inout_1,
                                    vecmem::vector<T2> &in_2,
                                    vecmem::vector<T3> &in_3,
                                    Arguments... args) {
        internal::offload_map(config, inout_1.size(), [&](int idx) mutable {
            algorithm.map(inout_1[idx], in_2[idx], in_3, args...);
        });
        return inout_1;
    }

    /// 3 iterable collections and no/default config
    template <class Algorithm, class R = typename Algorithm::result_t, typename T1,
            typename T2, typename T3,
            typename... Arguments,
            typename std::enable_if<std::is_base_of<
                    detail::parallel_mmap<T1, T2, T3, Arguments...>, Algorithm>::value>::type
            * = nullptr>
    vecmem::vector<R> &parallel_map(Algorithm &algorithm,
                                    __attribute__((unused))
                                    vecmem::memory_resource &mr,
                                    vecmem::vector<T1> &inout_1,
                                    vecmem::vector<T2> &in_2,
                                    vecmem::vector<T3> &in_3,
                                    Arguments... args) {

        return vecpar::omp::parallel_map(algorithm, mr, omp::getDefaultConfig(), inout_1, in_2, in_3, args...);
    }

    /// 2 iterable collections and config
    template <class Algorithm, class R = typename Algorithm::result_t, typename T1,
            typename T2,
            typename... Arguments,
            typename std::enable_if<std::is_base_of<
                    detail::parallel_map<R, T1, T2, Arguments...>, Algorithm>::value>::type
            * = nullptr>
    vecmem::vector<R> &parallel_map(Algorithm &algorithm,
                                    vecmem::memory_resource &mr,
                                    vecpar::config config, vecmem::vector<T1> &in_1,
                                    vecmem::vector<T2> &in_2,
                                    Arguments... args) {
        vecmem::vector<R> *map_result = new vecmem::vector<R>(in_1.size(), &mr);
        internal::offload_map(config, in_1.size(), [&](int idx) mutable {
            algorithm.map(map_result->at(idx), in_1[idx], in_2[idx], args...);
        });
        return *map_result;
    }

    /// 2 iterable collections and no/default config
    template <class Algorithm, class R = typename Algorithm::result_t, typename T1,
            typename T2,
            typename... Arguments,
            typename std::enable_if<std::is_base_of<
                    detail::parallel_map<T1, T2, Arguments...>, Algorithm>::value>::type
            * = nullptr>
    vecmem::vector<R> &parallel_map(Algorithm &algorithm,
                                    __attribute__((unused))
                                    vecmem::memory_resource &mr,
                                    vecmem::vector<T1> &in_1,
                                    vecmem::vector<T2> &in_2,
                                    Arguments... args) {
        return vecpar::omp::parallel_map(algorithm, mr, omp::getDefaultConfig(), in_1, in_2, args...);
    }

    /// 3 iterable collections and config
    template <class Algorithm, class R = typename Algorithm::result_t, typename T1,
            typename T2, typename T3,
            typename... Arguments,
            typename std::enable_if<std::is_base_of<
                    detail::parallel_map<R, T1, T2, T3, Arguments...>, Algorithm>::value>::type
            * = nullptr>
    vecmem::vector<R> &parallel_map(Algorithm &algorithm,
                                    vecmem::memory_resource &mr,
                                    vecpar::config config,
                                    vecmem::vector<T1> &in_1,
                                    vecmem::vector<T2> &in_2,
                                    vecmem::vector<T3> &in_3,
                                    Arguments... args) {
        vecmem::vector<R> *map_result = new vecmem::vector<R>(in_1.size(), &mr);
        internal::offload_map(config, in_1.size(), [&](int idx) mutable {
            algorithm.map(map_result->at(idx), in_1[idx], in_2[idx], in_3[idx], args...);
        });
        return *map_result;
    }

    /// 3 iterable collections and no/default config
    template <class Algorithm, class R = typename Algorithm::result_t, typename T1,
            typename T2, typename T3,
            typename... Arguments,
            typename std::enable_if<std::is_base_of<
                    detail::parallel_map<R, T1, T2, T3, Arguments...>, Algorithm>::value>::type
            * = nullptr>
    vecmem::vector<R> &parallel_map(Algorithm &algorithm,
                                    __attribute__((unused))
                                    vecmem::memory_resource &mr,
                                    vecmem::vector<T1> &in_1,
                                    vecmem::vector<T2> &in_2,
                                    vecmem::vector<T3> &in_3,
                                    Arguments... args) {

        return vecpar::omp::parallel_map(algorithm, mr, omp::getDefaultConfig(), in_1, in_2, in_3, args...);
    }
}
#endif //VECPAR_P_MAP_HPP
