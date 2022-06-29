#ifndef VECPAR_CUDA_PARALLELIZATION_HPP
#define VECPAR_CUDA_PARALLELIZATION_HPP

#include <cuda.h>
#include <type_traits>

#include "vecpar/core/algorithms/parallelizable_filter.hpp"
#include "vecpar/core/algorithms/parallelizable_map.hpp"
#include "vecpar/core/algorithms/parallelizable_map_filter.hpp"
#include "vecpar/core/algorithms/parallelizable_map_reduce.hpp"
#include "vecpar/core/algorithms/parallelizable_reduce.hpp"
#include "vecpar/core/definitions/config.hpp"

#include "vecpar/cuda/detail/common/config.hpp"
#include "vecpar/cuda/detail/common/cuda_utils.hpp"
#include "vecpar/cuda/detail/common/kernels.hpp"
#include "vecpar/cuda/detail/vecmem/host_memory.hpp"
#include "vecpar/cuda/detail/vecmem/managed_memory.hpp"

namespace vecpar::cuda {

template <typename Function, typename... Arguments>
void offload_map(vecpar::config config, size_t size, Function f,
                 Arguments... args) {
  kernel<<<config.m_gridSize, config.m_blockSize, config.m_memorySize>>>(
      size, f, args...);
  CHECK_ERROR(cudaGetLastError())
  CHECK_ERROR(cudaDeviceSynchronize())
}

template <typename Function, typename... Arguments>
void offload_map(size_t size, Function f, Arguments... args) {
  offload_map(vecpar::cuda::getDefaultConfig(size), size, f, args...);
}

template <typename Function, typename... Arguments>
void offload_reduce(size_t size, Function f, Arguments... args) {
  // TODO: improve performance by allocate the mutex on the device only
  int *lock; // all threads share on mutex.
  cudaMallocManaged((void **)&lock, sizeof(int));
  *lock = 0;
  vecpar::config config = vecpar::cuda::getReduceConfig<double>(size);
  vecpar::cuda::
      rkernel<<<config.m_gridSize, config.m_blockSize, config.m_memorySize>>>(
          lock, size, f, args...);
  CHECK_ERROR(cudaGetLastError())
  CHECK_ERROR(cudaDeviceSynchronize())
}

template <typename Function, typename... Arguments>
void parallel_map(vecpar::config config, size_t size, Function f,
                  Arguments... args) {
  offload_map(config, size, f, args...);
}

template <typename Function, typename... Arguments>
void parallel_map(size_t size, Function f, Arguments... args) {
  parallel_map(cuda::getDefaultConfig(size), size, f, args...);
}

template <typename Function, typename... Arguments>
void parallel_reduce(size_t size, Function f, Arguments... args) {
  offload_reduce(size, f, args...);
}

template <
    class MemoryResource, class Algorithm,
    class R = typename Algorithm::result_t, class T, typename... Arguments,
    typename std::enable_if_t<
        std::is_base_of<
            vecpar::algorithm::parallelizable_map_filter_1<R, T, Arguments...>,
            Algorithm>::value ||
            std::is_base_of<
                vecpar::algorithm::parallelizable_mmap_filter_1<T, Arguments...>,
                Algorithm>::value,
        bool> = true>
R &parallel_algorithm(Algorithm algorithm,
                      MemoryResource &mr,
                      vecpar::config config,
                      T &data,
                      Arguments... args) {

  return vecpar::cuda::parallel_map_filter<Algorithm, R, T, Arguments...>(
      algorithm, mr, config, data, args...);
}

template <
    class MemoryResource, class Algorithm,
    class R = typename Algorithm::result_t, class T, typename... Arguments,
    typename std::enable_if_t<
        std::is_base_of<
            vecpar::algorithm::parallelizable_map_filter_1<R, T, Arguments...>,
            Algorithm>::value ||
            std::is_base_of<
                vecpar::algorithm::parallelizable_mmap_filter_1<T, Arguments...>,
                Algorithm>::value,
        bool> = true>
R &parallel_algorithm(Algorithm algorithm, MemoryResource &mr,
                                      T &data,
                                      Arguments... args) {

  return vecpar::cuda::parallel_map_filter<Algorithm, R, T, Arguments...>(
      algorithm, mr, cuda::getDefaultConfig(data.size()), data, args...);
}

template <
    class MemoryResource, class Algorithm,
        typename R = typename Algorithm::intermediate_result_t,
        class Result = typename Algorithm::result_t,
    class T, typename... Arguments,
    typename std::enable_if_t<
        std::is_base_of<
            vecpar::algorithm::parallelizable_map_reduce_1<Result, R, T, Arguments...>,
            Algorithm>::value ||
            std::is_base_of<
                vecpar::algorithm::parallelizable_mmap_reduce_1<Result, T, Arguments...>,
                Algorithm>::value,
        bool> = true>
Result &parallel_algorithm(Algorithm algorithm, MemoryResource &mr,
                      vecpar::config config, T &data,
                      Arguments... args) {

  return vecpar::cuda::parallel_map_reduce<Algorithm, Result, R, T, Arguments...>(
      algorithm, mr, config, data, args...);
}

template <
    class MemoryResource, class Algorithm,
        typename R = typename Algorithm::intermediate_result_t,
        class Result = typename Algorithm::result_t,
    class T, typename... Arguments,
    typename std::enable_if_t<
        std::is_base_of<
            vecpar::algorithm::parallelizable_map_reduce_1<Result, R, T, Arguments...>,
            Algorithm>::value ||
            std::is_base_of<
                vecpar::algorithm::parallelizable_mmap_reduce_1<Result, T, Arguments...>,
                Algorithm>::value,
        bool> = true>
Result &parallel_algorithm(Algorithm algorithm, MemoryResource &mr,
                      T &data, Arguments... args) {

  return vecpar::cuda::parallel_map_reduce<Algorithm, Result, R, T, Arguments...>(
      algorithm, mr, cuda::getDefaultConfig(data.size()), data, args...);
}

} // namespace vecpar::cuda
#endif // VECPAR_CUDA_PARALLELIZATION_HPP
