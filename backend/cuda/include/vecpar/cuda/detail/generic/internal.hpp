#ifndef VECPAR_RAW_INTERNAL_HPP
#define VECPAR_RAW_INTERNAL_HPP

#include "vecpar/cuda/detail/common/config.hpp"
#include "vecpar/cuda/detail/common/cuda_utils.hpp"
#include "vecpar/cuda/detail/common/kernels.hpp"

namespace vecpar {

/// data structure which keeps information about
/// the converted vecmem::vectors to C-style pointers
template <typename T> struct cuda_data {
  T *ptr;
  size_t size;
};
} // namespace vecpar

namespace vecpar::cuda_raw {

template <typename Algorithm,
          typename R = typename Algorithm::intermediate_result_t,
          class T = typename Algorithm::input_t, typename... Arguments>
requires detail::is_mmap<Algorithm, T, Arguments...>
    cuda_data<typename T::value_type>
    parallel_map(Algorithm algorithm, vecpar::config config,
                 cuda_data<typename T::value_type> input, Arguments... args) {

  if (vecpar::config::isEmpty(config)) {
    config = vecpar::cuda::getDefaultConfig(input.size);
  }

  DEBUG_ACTION(printf("[MAP] nBlocks:%d, nThreads:%d, memorySize:%zu\n",
                      config.m_gridSize, config.m_blockSize,
                      config.m_memorySize);)

  vecpar::cuda::
      kernel<<<config.m_gridSize, config.m_blockSize, config.m_memorySize>>>(
          input.size,
          [=] __device__(int idx, Arguments... a) {
        algorithm.mapping_function(input.ptr[idx], a...);
          },
          args...);

  CHECK_ERROR(cudaGetLastError())
  CHECK_ERROR(cudaDeviceSynchronize())

  return input;
}

template <typename Algorithm,
          typename R = typename Algorithm::intermediate_result_t,
          class T = typename Algorithm::input_t, typename... Arguments>
requires detail::is_map<Algorithm, R, T, Arguments...>
    cuda_data<typename R::value_type>
    parallel_map(Algorithm algorithm, vecpar::config config,
                 cuda_data<typename T::value_type> input, Arguments... args) {

  if (vecpar::config::isEmpty(config)) {
    config = vecpar::cuda::getDefaultConfig(input.size);
  }

  DEBUG_ACTION(printf("[MAP] nBlocks:%d, nThreads:%d, memorySize:%zu\n",
                      config.m_gridSize, config.m_blockSize,
                      config.m_memorySize);)

  typename R::value_type *d_result = NULL;
  CHECK_ERROR(cudaMalloc((void **)&d_result,
                         input.size * sizeof(typename R::value_type)))

  vecpar::cuda::
      kernel<<<config.m_gridSize, config.m_blockSize, config.m_memorySize>>>(
          input.size,
          [=] __device__(int idx, Arguments... a) {
        algorithm.mapping_function(d_result[idx], input.ptr[idx], a...);
          },
          args...);

  CHECK_ERROR(cudaGetLastError())
  CHECK_ERROR(cudaDeviceSynchronize())

  return {d_result, input.size};
}

template <typename Algorithm, typename R>
void parallel_reduce(Algorithm algorithm, vecpar::config c,
                     cuda_data<typename R::value_type> result,
                     cuda_data<typename R::value_type> partial_result) {

  size_t size = partial_result.size;

  int *lock; // mutex.
  CHECK_ERROR(cudaMallocManaged((void **)&lock, sizeof(int)))
  *lock = 0;

  // make sure that an empty config ends up to be used
  if (vecpar::config::isEmpty(c)) {
    c = vecpar::cuda::getReduceConfig<typename R::value_type>(size);
  }

  DEBUG_ACTION(printf("[REDUCE] nBlocks:%d, nThreads:%d, memorySize:%zu\n",
                      c.m_gridSize, c.m_blockSize, c.m_memorySize);)

  vecpar::cuda::rkernel<<<c.m_gridSize, c.m_blockSize, c.m_memorySize>>>(
      lock, size, [=] __device__(int *lock) {
        extern __shared__ typename R::value_type temp[];

        size_t tid = threadIdx.x;
        temp[tid] = partial_result.ptr[tid + blockIdx.x * blockDim.x];
        size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
        //         printf("gidx %d starts with %f\n", gidx, temp[tid]);

        for (size_t d = blockDim.x >> 1; d >= 1; d >>= 1) {
          __syncthreads();

          /// for odd size and (larger) even number of threads, make sure
          /// we do not read from outside of the array
          bool within_array = ((tid + d) < blockDim.x) && (gidx + d < size);
          // printf("tid = %d, d = %d, gidx = %d, within_array? = %d (%d)\n",
          // tid, d, gidx,within_array, tid+d);

          if (tid < d && within_array) {
            //    printf("thread *%d*: read from index %d, %f + %f\n", gidx,
            //    tid+d, temp[tid], temp[tid+d]);
              algorithm.reducing_function(&temp[tid], temp[tid + d]);
          }
        }

        if (tid == 0) {
          do {
          } while (atomicCAS(lock, 0, 1)); // lock
          //     printf("thread %d: %f + %f \n", gidx, *result, temp[0]);
            algorithm.reducing_function(result.ptr, temp[0]);
          __threadfence();       // wait for write completion
          atomicCAS(lock, 1, 0); // release lock
        }
      });

  CHECK_ERROR(cudaGetLastError())
  CHECK_ERROR(cudaDeviceSynchronize())

  // release the memory allocated for the lock
  CHECK_ERROR(cudaFree(lock))
}

template <typename Algorithm, typename R>
void parallel_filter(Algorithm algorithm, vecpar::config c, int *idx,
                     cuda_data<R> d_result, cuda_data<R> partial_result) {

  size_t size = partial_result.size;

  int *lock; // mutex.
  CHECK_ERROR(cudaMallocManaged((void **)&lock, sizeof(int)))
  *lock = 0;

  // make sure that an empty config ends up to be used
  if (vecpar::config::isEmpty(c)) {
    c = vecpar::cuda::getReduceConfig<R>(size);
  }

  DEBUG_ACTION(printf("[FILTER] nBlocks:%d, nThreads:%d, memorySize:%zu\n",
                      c.m_gridSize, c.m_blockSize, c.m_memorySize);)

  vecpar::cuda::rkernel<<<c.m_gridSize, c.m_blockSize, c.m_memorySize>>>(
      lock, size,
      [=] __device__(int *lock, int *idx) {
        extern __shared__ R temp[];

        size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
        if (gidx > size)
          return;

        size_t tid = threadIdx.x;
        temp[tid] = partial_result.ptr[tid + blockIdx.x * blockDim.x];
        //    printf("thread %d loads element %f\n", gidx, temp[tid]);
        __syncthreads();

        if (tid == 0) {

          R temp_result[256]; // TODO: this should not be hardcoded;
          int count = 0;
          // printf("blocks = %d, threads = %d\n", c.gridSize, c.blockSize);
          for (size_t i = 0; i < static_cast<size_t>(c.m_blockSize) &&
                             tid + i < blockDim.x && gidx + i < size;
               i++) {
            if (algorithm.filtering_function(temp[tid + i])) {
              temp_result[count] = temp[tid + i];
              count++;
            }
          }

          int pos = 0; /// pos where to add in the global result

          do {
          } while (atomicCAS(lock, 0, 1)); // lock
          pos = *idx;
          atomicAdd(idx, count);
          __threadfence();
          atomicCAS(lock, 1, 0); // release lock

          // printf("element %f fits the list\n", temp[tid]);
          for (int i = 0; i < count; i++) {
            d_result.ptr[pos + i] = temp_result[i];
            //     printf("%f added to final\n", temp_result[i]);
          }
        }
      },
      idx);

  d_result.size = *idx;

  CHECK_ERROR(cudaGetLastError())
  CHECK_ERROR(cudaDeviceSynchronize())

  // release the memory allocated for the lock
  CHECK_ERROR(cudaFree(lock))
}
} // namespace vecpar::cuda_raw
#endif // VECPAR_RAW_INTERNAL_HPP
