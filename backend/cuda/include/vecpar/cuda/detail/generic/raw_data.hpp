#ifndef VECPAR_RAW_DATA_HPP
#define VECPAR_RAW_DATA_HPP

#include "vecpar/core/algorithms/parallelizable_filter.hpp"
#include "vecpar/core/algorithms/parallelizable_map.hpp"
#include "vecpar/core/algorithms/parallelizable_map_filter.hpp"
#include "vecpar/core/algorithms/parallelizable_map_reduce.hpp"
#include "vecpar/core/algorithms/parallelizable_reduce.hpp"

#include "vecpar/cuda/detail/generic/internal.hpp"

/// All input/output pointers refer to CUDA device memory only
namespace vecpar::cuda_raw {

template <class Algorithm, typename R, typename T, typename... Arguments>
cuda_data<typename R::value_type> parallel_map_filter(Algorithm algorithm, vecpar::config config,
                                 cuda_data<typename T::value_type> data, Arguments... args) {
  size_t size = data.size;

  cuda_data<typename R::value_type> map_result =
      cuda_raw::parallel_map<Algorithm, R, T, Arguments...>(algorithm, config, data, args...);

  typename R::value_type *result;
  CHECK_ERROR(cudaMalloc((void **)&result, size * sizeof(typename R::value_type)))

  cuda_data<typename R::value_type> result_filter{result, 0};

  int *idx; // global index
  CHECK_ERROR(cudaMallocManaged((void **)&idx, sizeof(int)))
  *idx = 0;
  cuda_raw::parallel_filter<Algorithm, typename R::value_type>(algorithm, config, idx, result_filter,
                                          map_result);

  result_filter.size = *idx;

  // release the memory allocated
  CHECK_ERROR(cudaFree(idx))
  CHECK_ERROR(cudaFree(data.ptr))

  return result_filter;
}

template <class Algorithm,
        class Result = typename Algorithm::result_t,
        typename R = typename Algorithm::intermediate_result_t,
        typename T, typename... Arguments>
cuda_data<Result> parallel_map_reduce(Algorithm algorithm,
                                      vecpar::config config,
                                      cuda_data<typename T::value_type> data,
                                      Arguments... args) {

  cuda_data<Result> map_result =
      cuda_raw::parallel_map<Algorithm, R, T, Arguments...>(algorithm, config, data, args...);

  Result *d_result;
  CHECK_ERROR(cudaMalloc((void **)&d_result, sizeof(Result)))
  CHECK_ERROR(cudaMemset(d_result, 0, sizeof(Result)))

  cuda_data<Result> result{d_result, 1};

  cuda_raw::parallel_reduce<Algorithm, R>(algorithm, config, result, map_result);
  // release allocate memory
  CHECK_ERROR(cudaFree(data.ptr))

  return result;
}

template <
    class Algorithm,
        typename Result = typename Algorithm::result_t,
        typename R = typename Algorithm::intermediate_result_t,
        class T = typename Algorithm::input_t,
    typename... Arguments,
    typename std::enable_if_t<
        std::is_base_of<
            vecpar::algorithm::parallelizable_map_reduce_1<Result, R, T, Arguments...>,
            Algorithm>::value ||
            std::is_base_of<
                vecpar::algorithm::parallelizable_mmap_reduce_1<Result, R, Arguments...>,
                Algorithm>::value,
        bool> = true>
cuda_data<Result> parallel_algorithm(Algorithm algorithm,
                                     vecpar::config config,
                                     cuda_data<typename T::value_type> data,
                                     Arguments... args) {

  return vecpar::cuda_raw::parallel_map_reduce<Algorithm,Result, R, T, Arguments...>(
      algorithm, config, data, args...);
}

template <
    class Algorithm, class R = typename Algorithm::result_t, class T,
    typename... Arguments,
    typename std::enable_if_t<
        std::is_base_of<
            vecpar::algorithm::parallelizable_map_filter_1<R, T, Arguments...>,
            Algorithm>::value ||
            std::is_base_of<
                vecpar::algorithm::parallelizable_mmap_filter_1<T, Arguments...>,
                Algorithm>::value,
        bool> = true>
cuda_data<typename R::value_type> parallel_algorithm(Algorithm algorithm, vecpar::config config,
                                cuda_data<typename T::value_type> data, Arguments... args) {

  return vecpar::cuda_raw::parallel_map_filter<Algorithm, R, T, Arguments...>(
      algorithm, config, data, args...);
}

template <class Algorithm, class R = typename Algorithm::result_t, class T,
          typename... Arguments,
          typename std::enable_if_t<
              std::is_base_of<
                  vecpar::algorithm::parallelizable_map_1<R, T, Arguments...>,
                  Algorithm>::value ||
                  std::is_base_of<
                      vecpar::algorithm::parallelizable_mmap_1<T, Arguments...>,
                      Algorithm>::value,
              bool> = true>
cuda_data<typename R::value_type> parallel_algorithm(Algorithm algorithm, vecpar::config config,
                                cuda_data<typename T::value_type> data, Arguments... args) {

  return vecpar::cuda_raw::parallel_map<Algorithm, R, T, Arguments...>(
      algorithm, config, data, args...);
}

template <class Algorithm, class T,
          typename std::enable_if_t<
              std::is_base_of<vecpar::algorithm::parallelizable_filter<T>,
                              Algorithm>::value,
              bool> = true>
cuda_data<T> parallel_algorithm(Algorithm algorithm, vecpar::config config,
                                cuda_data<typename T::value_type> data) {

  return vecpar::cuda_raw::parallel_filter<T>(algorithm, config, data);
}

template <class Algorithm, class R = typename Algorithm::result_t,
          typename std::enable_if_t<
              std::is_base_of<vecpar::algorithm::parallelizable_reduce<R>,
                              Algorithm>::value,
              bool> = true>
cuda_data<R> parallel_algorithm(Algorithm algorithm, vecpar::config config,
                                cuda_data<R> data) {

  return vecpar::cuda_raw::parallel_reduce<R>(algorithm, config, data);
}

template <class Algorithm, typename R>
cuda_data<R> copy_intermediate_result(vecmem::vector<R> &coll,
                                      cuda_data<R> mmap_result) {
  R *result_vec = (R *)malloc(mmap_result.size * sizeof(R));
  CHECK_ERROR(cudaMemcpy(result_vec, mmap_result.ptr,
                         mmap_result.size * sizeof(R), cudaMemcpyDeviceToHost));
  coll.clear();
  coll.assign(result_vec, result_vec + mmap_result.size);
  return mmap_result;
}
} // namespace vecpar::cuda_raw
#endif // VECPAR_RAW_DATA_HPP
