#ifndef VECPAR_MAIN_HPP
#define VECPAR_MAIN_HPP

#ifdef __CUDACC__
    #include "vecpar/cuda/cuda_parallelization.hpp"
#else

#include <vecmem/containers/vector.hpp>
#include "vecpar/omp/omp_parallelization.hpp"
#endif

namespace vecpar {

    template<class Algorithm,
            class MemoryResource,
            typename R = typename Algorithm::result_type,
            typename T,
            typename... Arguments>
    vecmem::vector<R>* parallel_map(Algorithm& algorithm,
                    MemoryResource& mr,
                    vecmem::vector<T>& data,
                    Arguments... args) {
#ifdef __CUDACC__
        return vecpar::cuda::parallel_map<Algorithm, R, T, Arguments...>(algorithm, mr, data, args...);
#else
        return vecpar::omp::parallel_map<Algorithm, R, T, Arguments...>(algorithm, mr, data, args...);
#endif
    }

    template<class Algorithm,
            class MemoryResource,
            typename R>
    R* parallel_reduce(Algorithm& algorithm,
                       MemoryResource& mr,
                       vecmem::vector<R>& data) {
#ifdef __CUDACC__
        return vecpar::cuda::parallel_reduce<Algorithm, R>(algorithm, mr, data);
#else
        return vecpar::omp::parallel_reduce<Algorithm, R>(algorithm, mr, data);
#endif
    }

    template<class Algorithm,
            class MemoryResource,
            typename T>
    vecmem::vector<T>* parallel_filter(Algorithm& algorithm,
                                        MemoryResource& mr,
                                        vecmem::vector<T>& data){
#ifdef __CUDACC__
        return vecpar::cuda::parallel_filter<Algorithm, T>(algorithm, mr, data);
#else
        return vecpar::omp::parallel_filter<Algorithm, T>(algorithm, mr, data);
#endif
    }

    template<class Algorithm,
            class MemoryResource,
            typename R = typename Algorithm::result_type,
            typename T,
            typename... Arguments>
    R* parallel_map_reduce(Algorithm& algorithm,
                          MemoryResource& mr,
                          vecmem::vector<T>& data,
                          Arguments... args) {
#ifdef __CUDACC__
        return vecpar::cuda::parallel_map_reduce<Algorithm, R, T, Arguments...>(algorithm, mr, data, args...);
#else
        return vecpar::omp::parallel_map_reduce<Algorithm, R, T, Arguments...>(algorithm, mr, data, args...);
#endif
    }

    template<class Algorithm,
            class MemoryResource,
            class R = typename Algorithm::result_type,
            typename T,
            typename... Arguments>
    vecmem::vector<R>* parallel_map_filter(Algorithm& algorithm,
                                          MemoryResource& mr,
                                          vecmem::vector<T>& data,
                                          Arguments... args) {
#ifdef __CUDACC__
        return vecpar::cuda::parallel_map_filter<Algorithm, R, T, Arguments...>(algorithm, mr, data, args...);
#else
        return vecpar::omp::parallel_map_filter<Algorithm, R, T, Arguments...>(algorithm, mr, data, args...);
#endif
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

        return parallel_map_filter(algorithm, mr, data, args...);
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

        return parallel_map_reduce(algorithm, mr, data, args...);
    }
}

#endif //VECPAR_MAIN_HPP
