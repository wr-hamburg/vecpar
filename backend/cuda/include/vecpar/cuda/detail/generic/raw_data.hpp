#ifndef VECPAR_RAW_DATA_HPP
#define VECPAR_RAW_DATA_HPP

#include "vecpar/core/algorithms/parallelizable_map.hpp"
#include "vecpar/core/algorithms/parallelizable_filter.hpp"
#include "vecpar/core/algorithms/parallelizable_reduce.hpp"
#include "vecpar/core/algorithms/parallelizable_map_reduce.hpp"
#include "vecpar/core/algorithms/parallelizable_map_filter.hpp"

#include "vecpar/cuda/detail/generic/internal.hpp"

/// All input/output pointers refer to CUDA device memory only
namespace vecpar::cuda_raw {

    template<class Algorithm,
            typename R,
            typename T,
            typename... Arguments>
    cuda_data<R> parallel_map_filter(Algorithm algorithm,
                          vecpar::config config,
                          cuda_data<T> data,
                          Arguments... args)  {
        size_t size = data.size;

        cuda_data<R> map_result = cuda_raw::parallel_map(algorithm,
                                           config,
                                           data,
                                            args...);

        R* result;
        CHECK_ERROR(cudaMalloc((void**)&result, size * sizeof(R)))

        cuda_data<R> result_filter{result, 0};

        int* idx; // global index
        CHECK_ERROR(cudaMallocManaged((void**)&idx, sizeof(int)))
        *idx = 0;
        cuda_raw::parallel_filter<Algorithm, R>(algorithm,
                                  config,
                                  idx,
                                  result_filter,
                                  map_result);

        result_filter.size = *idx;

        // release the memory allocated
        CHECK_ERROR(cudaFree(idx))
        CHECK_ERROR(cudaFree(data.ptr))

        return result_filter;
    }

    template<class Algorithm,
            typename R,
            typename T,
            typename... Arguments>
    cuda_data<R> parallel_map_reduce(Algorithm algorithm,
                           vecpar::config config,
                           cuda_data<T> data,
                           Arguments... args)  {

        cuda_data<R> map_result = cuda_raw::parallel_map(algorithm,
                                                  config,
                                                  data,
                                                  args...);

        R* d_result;
        CHECK_ERROR(cudaMalloc((void**)&d_result, sizeof(R)))
        CHECK_ERROR(cudaMemset(d_result, 0, sizeof(R)))

        cuda_data<R> result{d_result, 1};

        cuda_raw::parallel_reduce(algorithm,
                                  config,
                                  result,
                                  map_result);
        // release allocate memory
        CHECK_ERROR(cudaFree(data.ptr))

        return result;
    }

    template<class Algorithm,
             class R = typename Algorithm::result_t,
             class T, typename... Arguments,
             typename std::enable_if_t<std::is_base_of<vecpar::algorithm::parallelizable_map_reduce<R, T, Arguments...>, Algorithm>::value ||
                                       std::is_base_of<vecpar::algorithm::parallelizable_mmap_reduce<R, Arguments...>, Algorithm>::value, bool> = true>
    cuda_data<R> parallel_algorithm(Algorithm algorithm,
                          vecpar::config config,
                          cuda_data<T> data,
                          Arguments... args) {

        return vecpar::cuda_raw::parallel_map_reduce<Algorithm, R, T, Arguments...>(algorithm, config, data, args...);
    }

    template<class Algorithm,
            class R = typename Algorithm::result_t,
            class T, typename... Arguments,
            typename std::enable_if_t<std::is_base_of<vecpar::algorithm::parallelizable_map_filter<R, T, Arguments...>, Algorithm>::value ||
                                      std::is_base_of<vecpar::algorithm::parallelizable_mmap_filter<T, Arguments...>, Algorithm>::value, bool> = true>
    cuda_data<R> parallel_algorithm(Algorithm algorithm,
                                    vecpar::config config,
                                    cuda_data<T> data,
                                    Arguments... args) {

        return vecpar::cuda_raw::parallel_map_filter<Algorithm, R, T, Arguments...>(algorithm, config, data, args...);
    }

    template<class Algorithm,
            class R = typename Algorithm::result_t,
            class T, typename... Arguments,
            typename std::enable_if_t<std::is_base_of<vecpar::algorithm::parallelizable_map<R, T, Arguments...>, Algorithm>::value ||
                                      std::is_base_of<vecpar::algorithm::parallelizable_mmap<T, Arguments...>, Algorithm>::value, bool> = true>
    cuda_data<R> parallel_algorithm(Algorithm algorithm,
                                    vecpar::config config,
                                    cuda_data<T> data,
                                    Arguments... args) {

        return vecpar::cuda_raw::parallel_map<Algorithm, R, T, Arguments...>(algorithm, config, data, args...);
    }

    template<class Algorithm,
            class T,
            typename std::enable_if_t<std::is_base_of<vecpar::algorithm::parallelizable_filter<T>, Algorithm>::value, bool> = true>
    cuda_data<T> parallel_algorithm(Algorithm algorithm,
                                    vecpar::config config,
                                    cuda_data<T> data) {

        return vecpar::cuda_raw::parallel_filter<T>(algorithm, config, data);
    }

    template<class Algorithm,
            class R = typename Algorithm::result_t,
            typename std::enable_if_t<std::is_base_of<vecpar::algorithm::parallelizable_reduce<R>, Algorithm>::value, bool> = true>
    cuda_data<R> parallel_algorithm(Algorithm algorithm,
                                    vecpar::config config,
                                    cuda_data<R> data) {

        return vecpar::cuda_raw::parallel_reduce<R>(algorithm, config, data);
    }

    template<class Algorithm,
            typename R,
            typename... Arguments>
    cuda_data<R> copy_intermediate_result(vecmem::vector<R>& coll, cuda_data<R> mmap_result) {
        R* result_vec = (R*) malloc(mmap_result.size * sizeof(R));
        CHECK_ERROR(cudaMemcpy(result_vec, mmap_result.ptr, mmap_result.size * sizeof(R), cudaMemcpyDeviceToHost));
        coll.clear();
        coll.assign(result_vec, result_vec + mmap_result.size);
        return mmap_result;
    }
}
#endif //VECPAR_RAW_DATA_HPP
