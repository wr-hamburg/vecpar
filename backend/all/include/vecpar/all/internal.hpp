#ifndef VECPAR_INTERNAL_HPP
#define VECPAR_INTERNAL_HPP

#include <vecmem/containers/vector.hpp>

#if defined(__CUDA__) && defined(__clang__)
    #include "vecpar/cuda/cuda_parallelization.hpp"
#endif

#if defined(_OPENMP)
    #include "vecpar/omp/omp_parallelization.hpp"
#endif

namespace vecpar {

    template<class Algorithm,
            class MemoryResource,
            typename R = typename Algorithm::result_t,
            typename T,
            typename... Arguments>
    vecmem::vector<R> &parallel_map(Algorithm &algorithm,
                                    MemoryResource &mr,
                                    vecpar::config config,
                                    vecmem::vector<T> &data,
                                    Arguments... args) {
#if defined(__CUDA__) && defined(__clang__)
        return vecpar::cuda::parallel_map<Algorithm, R, T, Arguments...>(algorithm, mr, config, data, args...);
#elif defined(_OPENMP)
        return vecpar::omp::parallel_map<Algorithm, R, T, Arguments...>(algorithm, mr, config, data, args...);
#endif
    }

    template<class Algorithm,
            class MemoryResource,
            typename R = typename Algorithm::result_t,
            typename T,
            typename... Arguments>
    vecmem::vector<R> &parallel_map(Algorithm &algorithm,
                                    MemoryResource &mr,
                                    vecmem::vector<T> &data,
                                    Arguments... args) {
#if defined(__CUDA__) && defined(__clang__)
        return vecpar::cuda::parallel_map<Algorithm, R, T, Arguments...>(algorithm, mr, data, args...);
#elif defined(_OPENMP)
        return vecpar::omp::parallel_map<Algorithm, R, T, Arguments...>(algorithm, mr, data, args...);
#endif
    }

    template<class Algorithm,
            class MemoryResource,
            typename R>
    R &parallel_reduce(Algorithm &algorithm,
                       MemoryResource &mr,
                       vecmem::vector<R> &data) {
#if defined(__CUDA__) && defined(__clang__)
        return vecpar::cuda::parallel_reduce<Algorithm, R>(algorithm, mr, data);
#elif defined(_OPENMP)
        return vecpar::omp::parallel_reduce<Algorithm, R>(algorithm, mr, data);
#endif
    }

    template<class Algorithm,
            class MemoryResource,
            typename T>
    vecmem::vector<T> &parallel_filter(Algorithm &algorithm,
                                       MemoryResource &mr,
                                       vecmem::vector<T> &data) {
#if defined(__CUDA__) && defined(__clang__)
        return vecpar::cuda::parallel_filter<Algorithm, T>(algorithm, mr, data);
#elif defined(_OPENMP)
        return vecpar::omp::parallel_filter<Algorithm, T>(algorithm, mr, data);
#endif
    }

    template<class Algorithm,
            class MemoryResource,
            typename R = typename Algorithm::result_t,
            typename T,
            typename... Arguments>
    R &parallel_map_reduce(Algorithm &algorithm,
                           MemoryResource &mr,
                           vecpar::config config,
                           vecmem::vector<T> &data,
                           Arguments... args) {
#if defined(__CUDA__) && defined(__clang__)
        return vecpar::cuda::parallel_map_reduce<Algorithm, R, T, Arguments...>(algorithm, mr, config, data, args...);
#elif defined(_OPENMP)
        return vecpar::omp::parallel_map_reduce<Algorithm, R, T, Arguments...>(
                    algorithm, mr, config, data, args...);
#endif
    }

    template<class Algorithm,
            class MemoryResource,
            typename R = typename Algorithm::result_t,
            typename T,
            typename... Arguments>
    R &parallel_map_reduce(Algorithm &algorithm,
                           MemoryResource &mr,
                           vecmem::vector<T> &data,
                           Arguments... args) {
#if defined(__CUDA__) && defined(__clang__)
        return vecpar::cuda::parallel_map_reduce<Algorithm, R, T, Arguments...>(algorithm, mr, data, args...);
#elif defined(_OPENMP)
        return vecpar::omp::parallel_map_reduce<Algorithm, R, T, Arguments...>(
                    algorithm, mr, data, args...);
#endif
    }

    template<class Algorithm,
            class MemoryResource,
            class R = typename Algorithm::result_t,
            typename T,
            typename... Arguments>
    vecmem::vector<R> &parallel_map_filter(Algorithm &algorithm,
                                           MemoryResource &mr,
                                           vecpar::config config,
                                           vecmem::vector<T> &data,
                                           Arguments... args) {
#if defined(__CUDA__) && defined(__clang__)
        return vecpar::cuda::parallel_map_filter<Algorithm, R, T, Arguments...>(algorithm, mr, config, data, args...);
#elif defined(_OPENMP)
        return vecpar::omp::parallel_map_filter<Algorithm, R, T, Arguments...>(
                    algorithm, mr, config, data, args...);
#endif
    }

    template<class Algorithm,
            class MemoryResource,
            class R = typename Algorithm::result_t,
            typename T,
            typename... Arguments>
    vecmem::vector<R> &parallel_map_filter(Algorithm &algorithm,
                                           MemoryResource &mr,
                                           vecmem::vector<T> &data,
                                           Arguments... args) {
#if defined(__CUDA__) && defined(__clang__)
        return vecpar::cuda::parallel_map_filter<Algorithm, R, T, Arguments...>(algorithm, mr, data, args...);
#elif defined(_OPENMP)
        return vecpar::omp::parallel_map_filter<Algorithm, R, T, Arguments...>(
                    algorithm, mr, data, args...);
#endif
    }
} // end namespace

#endif //VECPAR_INTERNAL_HPP
