#ifndef VECPAR_CUDA_INTERNAL_HPP
#define VECPAR_CUDA_INTERNAL_HPP

#include <functional>

#include <vecmem/containers/data/vector_view.hpp>
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/containers/jagged_device_vector.hpp>
#include <vecmem/containers/jagged_vector.hpp>
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/utils/cuda/copy.hpp>

#include "vecpar/core/definitions/common.hpp"
#include "vecpar/core/definitions/config.hpp"
#include "vecpar/core/definitions/helper.hpp"
#include "vecpar/core/definitions/types.hpp"
#include "vecpar/cuda/detail/common/config.hpp"
#include "vecpar/cuda/detail/common/cuda_utils.hpp"
#include "vecpar/cuda/detail/common/kernels.hpp"

using namespace std::placeholders;

namespace helper {

template <Jagged_vector_type T>
__device__ vecmem::jagged_device_vector<typename T::value_type::value_type>
get_device_container(auto in) {
  vecmem::jagged_device_vector<typename T::value_type::value_type> dv_data(in);
  return dv_data;
}

template <typename T>
__device__ vecmem::device_vector<typename T::value_type>
get_device_container(auto in) {
  vecmem::device_vector<typename T::value_type> dv_data(in);
  return dv_data;
}

template <typename T>
vecmem::data::jagged_vector_view<value_type_t<T>>
get_view(vecmem::data::jagged_vector_data<value_type_t<T>> &coll) {
  //  std::cout << "jagged" << std::endl;
  return vecmem::get_data(coll);
}

template <typename T>
vecmem::data::jagged_vector_view<value_type_t<T>>
get_view(vecmem::data::jagged_vector_view<value_type_t<T>> &coll) {
  //  std::cout << "jagged" << std::endl;
  return coll;
}

template <typename T>
vecmem::data::vector_view<value_type_t<T>>
get_view(vecmem::data::vector_view<value_type_t<T>> &coll) {
  //  std::cout << "not jagged" << std::endl;
  return coll;
}

template <typename T>
vecmem::data::vector_view<value_type_t<T>>
get_view(vecmem::data::vector_buffer<value_type_t<T>> &coll) {
  return vecmem::get_data(coll);
}

template <typename T>
vecmem::data::jagged_vector_view<value_type_t<T>>
get_view(vecmem::data::jagged_vector_buffer<value_type_t<T>> &coll) {
  return vecmem::get_data(coll);
}
} // namespace helper

namespace internal {

static vecmem::cuda::device_memory_resource d_mem;
static vecmem::cuda::copy copy;

template <typename Algorithm, typename R = typename Algorithm::result_t,
          typename T, typename... Arguments>
requires vecpar::detail::is_map_1<Algorithm, R, T, Arguments...>
void parallel_map(vecpar::config c, size_t size, Algorithm algorithm,
                  auto &result, auto &data, Arguments &...args) {

  // make sure that an empty config doesn't end up to be used
  if (vecpar::config::isEmpty(c)) {
    c = vecpar::cuda::getDefaultConfig(size);
  }

  // an extra call is needed to get the data when the collection is a jagged one
  auto result_view = helper::get_view<R>(result);
  auto data_view = helper::get_view<T>(data);

  DEBUG_ACTION(printf("[MAP] nBlocks:%d, nThreads:%d, memorySize:%zu\n",
                      c.m_gridSize, c.m_blockSize, c.m_memorySize);)

  vecpar::cuda::kernel<<<c.m_gridSize, c.m_blockSize, c.m_memorySize>>>(
      size,
      [algorithm] __device__(int idx, auto &d_result, const auto &d_in,
                             Arguments... a) {
        auto dv_data = helper::get_device_container<T>(d_in);
        auto dv_result = helper::get_device_container<R>(d_result);
        //     printf("[mapper] data[%d]=%f\n", idx, dv_data[idx]);
        algorithm.map(dv_result[idx], dv_data[idx], a...);
        //   printf("[mapper] result[%d]=%f\n", idx, dv_result[idx]);
      },
      result_view, data_view, args...);

  CHECK_ERROR(cudaGetLastError())
  CHECK_ERROR(cudaDeviceSynchronize())
}

template <typename Algorithm, typename R = typename Algorithm::result_t,
          typename T1, typename T2, typename... Arguments>
requires vecpar::detail::is_map_2<Algorithm, R, T1, T2, Arguments...>
void parallel_map(vecpar::config c, size_t size, Algorithm algorithm,
                  auto &result, auto &in_1, auto &in_2, Arguments &...args) {

  // make sure that an empty config doesn't end up to be used
  if (vecpar::config::isEmpty(c)) {
    c = vecpar::cuda::getDefaultConfig(size);
  }

  DEBUG_ACTION(printf("[MAP] nBlocks:%d, nThreads:%d, memorySize:%zu\n",
                      c.m_gridSize, c.m_blockSize, c.m_memorySize);)

  // an extra call is needed to get the data when the collection is a jagged one
  auto result_view = helper::get_view<R>(result);
  auto in_1_view = helper::get_view<T1>(in_1);
  auto in_2_view = helper::get_view<T2>(in_2);

  vecpar::cuda::kernel<<<c.m_gridSize, c.m_blockSize, c.m_memorySize>>>(
      size,
      [algorithm] __device__(int idx, auto &d_result, const auto &d_in_1,
                             const auto &d_in_2, Arguments... a) {
        auto dv_data_1 = helper::get_device_container<T1>(d_in_1);
        auto dv_data_2 = helper::get_device_container<T2>(d_in_2);
        auto dv_result = helper::get_device_container<R>(d_result);
        //     printf("[mapper] data[%d]=%f\n", idx, dv_data[idx]);
        algorithm.map(dv_result[idx], dv_data_1[idx], dv_data_2[idx], a...);
        //   printf("[mapper] result[%d]=%f\n", idx, dv_result[idx]);
      },
      result_view, in_1_view, in_2_view, args...);

  CHECK_ERROR(cudaGetLastError())
  CHECK_ERROR(cudaDeviceSynchronize())
}

template <typename Algorithm, typename R = typename Algorithm::result_t,
          typename T1, typename T2, typename T3, typename... Arguments>
requires vecpar::detail::is_map_3<Algorithm, R, T1, T2, T3, Arguments...>
void parallel_map(vecpar::config c, size_t size, Algorithm algorithm,
                  auto &result, auto &in_1, auto &in_2, auto &in_3,
                  Arguments &...args) {

  // make sure that an empty config doesn't end up to be used
  if (vecpar::config::isEmpty(c)) {
    c = vecpar::cuda::getDefaultConfig(size);
  }

  DEBUG_ACTION(printf("[MAP] nBlocks:%d, nThreads:%d, memorySize:%zu\n",
                      c.m_gridSize, c.m_blockSize, c.m_memorySize);)
  auto result_view = helper::get_view<R>(result);
  auto in_1_view = helper::get_view<T1>(in_1);
  auto in_2_view = helper::get_view<T2>(in_2);
  auto in_3_view = helper::get_view<T3>(in_3);

  vecpar::cuda::kernel<<<c.m_gridSize, c.m_blockSize, c.m_memorySize>>>(
      size,
      [algorithm] __device__(int idx, auto &d_result, const auto &d_in_1,
                             const auto &d_in_2, const auto &d_in_3,
                             Arguments... a) {
        auto dv_data_1 = helper::get_device_container<T1>(d_in_1);
        auto dv_data_2 = helper::get_device_container<T2>(d_in_2);
        auto dv_data_3 = helper::get_device_container<T3>(d_in_3);
        auto dv_result = helper::get_device_container<R>(d_result);
        //       printf("[mapper] data[%d]=%f\n", idx, dv_data_3[idx]);
        algorithm.map(dv_result[idx], dv_data_1[idx], dv_data_2[idx],
                      dv_data_3[idx], a...);
        //   printf("[mapper] result[%d]=%f\n", idx, dv_result[idx]);
      },
      result_view, in_1_view, in_2_view, in_3_view, args...);

  CHECK_ERROR(cudaGetLastError())
  CHECK_ERROR(cudaDeviceSynchronize())
}

template <typename Algorithm, typename R = typename Algorithm::result_t,
          typename T1, typename T2, typename T3, typename T4,
          typename... Arguments>
requires vecpar::detail::is_map_4<Algorithm, R, T1, T2, T3, T4, Arguments...>
void parallel_map(vecpar::config c, size_t size, Algorithm algorithm,
                  auto &result, auto &in_1, auto &in_2, auto &in_3, auto &in_4,
                  Arguments &...args) {

  // make sure that an empty config doesn't end up to be used
  if (vecpar::config::isEmpty(c)) {
    c = vecpar::cuda::getDefaultConfig(size);
  }

  DEBUG_ACTION(printf("[MAP] nBlocks:%d, nThreads:%d, memorySize:%zu\n",
                      c.m_gridSize, c.m_blockSize, c.m_memorySize);)
  auto result_view = helper::get_view<R>(result);
  auto in_1_view = helper::get_view<T1>(in_1);
  auto in_2_view = helper::get_view<T2>(in_2);
  auto in_3_view = helper::get_view<T3>(in_3);
  auto in_4_view = helper::get_view<T4>(in_4);

  vecpar::cuda::kernel<<<c.m_gridSize, c.m_blockSize, c.m_memorySize>>>(
      size,
      [algorithm] __device__(int idx, auto &d_result, const auto &d_in_1,
                             const auto &d_in_2, const auto &d_in_3,
                             const auto &d_in_4, Arguments... a) {
        auto dv_data_1 = helper::get_device_container<T1>(d_in_1);
        auto dv_data_2 = helper::get_device_container<T2>(d_in_2);
        auto dv_data_3 = helper::get_device_container<T3>(d_in_3);
        auto dv_data_4 = helper::get_device_container<T4>(d_in_4);
        auto dv_result = helper::get_device_container<R>(d_result);
        //     printf("[mapper] data[%d]=%f\n", idx, dv_data[idx]);
        algorithm.map(dv_result[idx], dv_data_1[idx], dv_data_2[idx],
                      dv_data_3[idx], dv_data_4[idx], a...);
        //   printf("[mapper] result[%d]=%f\n", idx, dv_result[idx]);
      },
      result_view, in_1_view, in_2_view, in_3_view, in_4_view, args...);

  CHECK_ERROR(cudaGetLastError())
  CHECK_ERROR(cudaDeviceSynchronize())
}

template <typename Algorithm, typename R = typename Algorithm::result_t,
          typename T1, typename T2, typename T3, typename T4, typename T5,
          typename... Arguments>
requires vecpar::detail::is_map_5<Algorithm, R, T1, T2, T3, T4, T5,
                                  Arguments...>
void parallel_map(vecpar::config c, size_t size, Algorithm algorithm,
                  auto &result, auto &in_1, auto &in_2, auto &in_3, auto &in_4,
                  auto &in_5, Arguments &...args) {

  // make sure that an empty config doesn't end up to be used
  if (vecpar::config::isEmpty(c)) {
    c = vecpar::cuda::getDefaultConfig(size);
  }

  DEBUG_ACTION(printf("[MAP] nBlocks:%d, nThreads:%d, memorySize:%zu\n",
                      c.m_gridSize, c.m_blockSize, c.m_memorySize);)
  auto result_view = helper::get_view<R>(result);
  auto in_1_view = helper::get_view<T1>(in_1);
  auto in_2_view = helper::get_view<T2>(in_2);
  auto in_3_view = helper::get_view<T3>(in_3);
  auto in_4_view = helper::get_view<T4>(in_4);
  auto in_5_view = helper::get_view<T5>(in_5);

  vecpar::cuda::kernel<<<c.m_gridSize, c.m_blockSize, c.m_memorySize>>>(
      size,
      [algorithm] __device__(int idx, auto &d_result, const auto &d_in_1,
                             const auto &d_in_2, const auto &d_in_3,
                             const auto &d_in_4, const auto &d_in_5,
                             Arguments... a) {
        auto dv_data_1 = helper::get_device_container<T1>(d_in_1);
        auto dv_data_2 = helper::get_device_container<T2>(d_in_2);
        auto dv_data_3 = helper::get_device_container<T3>(d_in_3);
        auto dv_data_4 = helper::get_device_container<T4>(d_in_4);
        auto dv_data_5 = helper::get_device_container<T5>(d_in_5);
        auto dv_result = helper::get_device_container<R>(d_result);
        //     printf("[mapper] data[%d]=%f\n", idx, dv_data[idx]);
        algorithm.map(dv_result[idx], dv_data_1[idx], dv_data_2[idx],
                      dv_data_3[idx], dv_data_4[idx], dv_data_5[idx], a...);
        //   printf("[mapper] result[%d]=%f\n", idx, dv_result[idx]);
      },
      result_view, in_1_view, in_2_view, in_3_view, in_4_view, in_5_view,
      args...);

  CHECK_ERROR(cudaGetLastError())
  CHECK_ERROR(cudaDeviceSynchronize())
}

template <typename Algorithm, typename TT, typename... Arguments>
requires vecpar::detail::is_mmap_1<Algorithm, TT, Arguments...>
void parallel_mmap(vecpar::config c, size_t size, const Algorithm algorithm,
                   auto &input_output, Arguments &...args) {

 // using func_t = typename Algorithm::func_t;

  // make sure that an empty config doesn't end up to be used
  if (vecpar::config::isEmpty(c)) {
    c = vecpar::cuda::getDefaultConfig(size);
  }

  DEBUG_ACTION(printf("[MAP] nBlocks:%d, nThreads:%d, memorySize:%zu\n",
                      c.m_gridSize, c.m_blockSize, c.m_memorySize);)

  // an extra call is needed to get the data when the collection is a jagged one
  auto input_output_view = helper::get_view<TT>(input_output);
  vecpar::cuda::kernel<<<c.m_gridSize, c.m_blockSize, c.m_memorySize>>>(
                    size,
            [algorithm] __device__(int idx, auto &d_in_out_view, Arguments... a) {
                auto dv_data = helper::get_device_container<TT>(d_in_out_view);
                //   printf("[mapper] data[%d]=%f\n", idx, dv_data[idx]);
                algorithm.map(dv_data[idx], a...);
                //     printf("[mapper] result[%d]=%f\n", idx, dv_data[idx]);
            },
            input_output_view, args...);
        CHECK_ERROR(cudaGetLastError())
        CHECK_ERROR(cudaDeviceSynchronize())
}

template <typename Algorithm, typename T1, typename T2, typename... Arguments>
requires vecpar::detail::is_mmap_2<Algorithm, T1, T2, Arguments...>
void parallel_mmap(vecpar::config c, size_t size, Algorithm algorithm,
                   auto &input_output, auto &in_2, Arguments &...args) {

  // make sure that an empty config doesn't end up to be used
  if (vecpar::config::isEmpty(c)) {
    c = vecpar::cuda::getDefaultConfig(size);
  }

  DEBUG_ACTION(printf("[MAP] nBlocks:%d, nThreads:%d, memorySize:%zu\n",
                      c.m_gridSize, c.m_blockSize, c.m_memorySize);)

  // an extra call is needed to get the data when the collection is a jagged one
  auto input_output_view = helper::get_view<T1>(input_output);
  auto in_2_view = helper::get_view<T2>(in_2);

  vecpar::cuda::kernel<<<c.m_gridSize, c.m_blockSize, c.m_memorySize>>>(
      size,
      [algorithm] __device__(int idx, auto &d_in_out, const auto &d_in_2,
                             Arguments... a) {
        auto dv_data_1 = helper::get_device_container<T1>(d_in_out);
        auto dv_data_2 = helper::get_device_container<T2>(d_in_2);
        algorithm.map(dv_data_1[idx], dv_data_2[idx], a...);
      },
      input_output_view, in_2_view, args...);

  CHECK_ERROR(cudaGetLastError())
  CHECK_ERROR(cudaDeviceSynchronize())
}

template <typename Algorithm, typename T1, typename T2, typename T3,
          typename... Arguments>
requires vecpar::detail::is_mmap_3<Algorithm, T1, T2, T3, Arguments...>
void parallel_mmap(vecpar::config c, size_t size, Algorithm algorithm,
                   auto &input_output, auto &in_2, auto &in_3,
                   Arguments &...args) {

  // make sure that an empty config doesn't end up to be used
  if (vecpar::config::isEmpty(c)) {
    c = vecpar::cuda::getDefaultConfig(size);
  }

  DEBUG_ACTION(printf("[MAP] nBlocks:%d, nThreads:%d, memorySize:%zu\n",
                      c.m_gridSize, c.m_blockSize, c.m_memorySize);)
  // an extra call is needed to get the data when the collection is a jagged one
  auto input_output_view = helper::get_view<T1>(input_output);
  auto in_2_view = helper::get_view<T2>(in_2);
  auto in_3_view = helper::get_view<T3>(in_3);

  vecpar::cuda::kernel<<<c.m_gridSize, c.m_blockSize, c.m_memorySize>>>(
      size,
      [algorithm] __device__(int idx, auto &d_in_out, const auto &d_in_2,
                             const auto &d_in_3, Arguments... a) {
        auto dv_data_1 = helper::get_device_container<T1>(d_in_out);
        auto dv_data_2 = helper::get_device_container<T2>(d_in_2);
        auto dv_data_3 = helper::get_device_container<T3>(d_in_3);

        algorithm.map(dv_data_1[idx], dv_data_2[idx], dv_data_3[idx], a...);
      },
      input_output_view, in_2_view, in_3_view, args...);

  CHECK_ERROR(cudaGetLastError())
  CHECK_ERROR(cudaDeviceSynchronize())
}

template <typename Algorithm, typename T1, typename T2, typename T3,
          typename T4, typename... Arguments>
requires vecpar::detail::is_mmap_4<Algorithm, T1, T2, T3, T4, Arguments...>
void parallel_mmap(vecpar::config c, size_t size, Algorithm algorithm,
                   auto &input_output, auto &in_2, auto &in_3, auto &in_4,
                   Arguments &...args) {

  // make sure that an empty config doesn't end up to be used
  if (vecpar::config::isEmpty(c)) {
    c = vecpar::cuda::getDefaultConfig(size);
  }
  // an extra call is needed to get the data when the collection is a jagged one
  auto input_output_view = helper::get_view<T1>(input_output);
  auto in_2_view = helper::get_view<T2>(in_2);
  auto in_3_view = helper::get_view<T3>(in_3);
  auto in_4_view = helper::get_view<T4>(in_4);

  DEBUG_ACTION(printf("[MAP] nBlocks:%d, nThreads:%d, memorySize:%zu\n",
                      c.m_gridSize, c.m_blockSize, c.m_memorySize);)

  vecpar::cuda::kernel<<<c.m_gridSize, c.m_blockSize, c.m_memorySize>>>(
      size,
      [algorithm] __device__(int idx, auto &d_in_out, const auto &d_in_2,
                             const auto &d_in_3, const auto &d_in_4,
                             Arguments... a) {
        auto dv_data_1 = helper::get_device_container<T1>(d_in_out);
        auto dv_data_2 = helper::get_device_container<T2>(d_in_2);
        auto dv_data_3 = helper::get_device_container<T3>(d_in_3);
        auto dv_data_4 = helper::get_device_container<T4>(d_in_4);

        algorithm.map(dv_data_1[idx], dv_data_2[idx], dv_data_3[idx],
                      dv_data_4[idx], a...);
      },
      input_output_view, in_2_view, in_3_view, in_4_view, args...);

  CHECK_ERROR(cudaGetLastError())
  CHECK_ERROR(cudaDeviceSynchronize())
}

template <typename Algorithm, typename T1, typename T2, typename T3,
          typename T4, typename T5, typename... Arguments>
requires vecpar::detail::is_mmap_5<Algorithm, T1, T2, T3, T4, T5, Arguments...>
void parallel_mmap(vecpar::config c, size_t size, Algorithm algorithm,
                   auto &input_output, auto &in_2, auto &in_3, auto &in_4,
                   auto &in_5, Arguments &...args) {

  // make sure that an empty config doesn't end up to be used
  if (vecpar::config::isEmpty(c)) {
    c = vecpar::cuda::getDefaultConfig(size);
  }

  DEBUG_ACTION(printf("[MAP] nBlocks:%d, nThreads:%d, memorySize:%zu\n",
                      c.m_gridSize, c.m_blockSize, c.m_memorySize);)

  // an extra call is needed to get the data when the collection is a jagged one
  auto input_output_view = helper::get_view<T1>(input_output);
  auto in_2_view = helper::get_view<T2>(in_2);
  auto in_3_view = helper::get_view<T3>(in_3);
  auto in_4_view = helper::get_view<T4>(in_4);
  auto in_5_view = helper::get_view<T5>(in_5);

  vecpar::cuda::kernel<<<c.m_gridSize, c.m_blockSize, c.m_memorySize>>>(
      size,
      [algorithm] __device__(int idx, auto &view_1, auto &view_2, auto &view_3,
                             auto &view_4, auto &view_5, Arguments... a) {
        auto dv_data_1 = helper::get_device_container<T1>(view_1);
        auto dv_data_2 = helper::get_device_container<T2>(view_2);
        auto dv_data_3 = helper::get_device_container<T3>(view_3);
        auto dv_data_4 = helper::get_device_container<T4>(view_4);
        auto dv_data_5 = helper::get_device_container<T5>(view_5);

        algorithm.map(dv_data_1[idx], dv_data_2[idx], dv_data_3[idx],
                      dv_data_4[idx], dv_data_5[idx], a...);
      },
      input_output_view, in_2_view, in_3_view, in_4_view, in_5_view, args...);

  CHECK_ERROR(cudaGetLastError())
  CHECK_ERROR(cudaDeviceSynchronize())
}

/// based on
/// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf more
/// efficient versions using thread-block parametrization can be implemented
template <typename Algorithm, typename R>
void parallel_reduce(vecpar::config c, size_t size, Algorithm algorithm,
                     R *result,
                     vecmem::data::vector_view<R> partial_result_view) {

  int *lock; // mutex.
  CHECK_ERROR(cudaMallocManaged((void **)&lock, sizeof(int)))
  *lock = 0;

  // make sure that an empty config ends up to be used
  if (vecpar::config::isEmpty(c)) {
    c = vecpar::cuda::getReduceConfig<R>(size);
  }

  DEBUG_ACTION(printf("[REDUCE] nBlocks:%d, nThreads:%d, memorySize:%zu\n",
                      c.m_gridSize, c.m_blockSize, c.m_memorySize);)

  vecpar::cuda::rkernel<<<c.m_gridSize, c.m_blockSize, c.m_memorySize>>>(
      lock, size, [=] __device__(int *lock) {
        vecmem::device_vector<R> partial_result(partial_result_view);

        extern __shared__ char smem[];
        R* temp = reinterpret_cast<R*>(smem);
        //extern __shared__ R temp[];

        size_t tid = threadIdx.x;
        temp[tid] = partial_result[tid + blockIdx.x * blockDim.x];
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
            algorithm.reduce(&temp[tid], temp[tid + d]);
          }
        }

        if (tid == 0) {
          do {
          } while (atomicCAS(lock, 0, 1)); // lock
          //     printf("thread %d: %f + %f \n", gidx, *result, temp[0]);
          algorithm.reduce(result, temp[0]);
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
void parallel_reduce(size_t size, Algorithm algorithm, R *result,
                     vecmem::data::vector_view<R> partial_result) {

  parallel_reduce<Algorithm, R>(vecpar::cuda::getReduceConfig<R>(size), size, algorithm,
                  result, partial_result);
}

template <typename Algorithm, typename R>
void parallel_filter(vecpar::config c, size_t size, Algorithm algorithm,
                     int *idx, vecmem::data::vector_view<R> &result_view,
                     vecmem::data::vector_view<R> partial_result_view) {

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
        vecmem::device_vector<R> d_result(result_view);
        vecmem::device_vector<R> partial_result(partial_result_view);

        extern __shared__ char smem[];
        R* temp = reinterpret_cast<R*>(smem);
       // extern __shared__ R temp[];

        size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
        if (gidx > size)
          return;

        size_t tid = threadIdx.x;
        temp[tid] = partial_result[tid + blockIdx.x * blockDim.x];
        __syncthreads();

        if (tid == 0) {
          R temp_result[256]; // TODO: this should not be hardcoded;
          int count = 0;
          // printf("blocks = %d, threads = %d\n", c.gridSize, c.blockSize);
          for (size_t i = 0; i < static_cast<size_t>(c.m_blockSize) &&
                             tid + i < blockDim.x && gidx + i < size;
               i++) {
            if (algorithm.filter(temp[tid + i])) {
              temp_result[count] = temp[tid + i];
              count++;
              //            printf("%f added to temp\n", temp[tid+i]);
            }
          }

          int pos = 0; /// pos where to add in the global result

          do {
          } while (atomicCAS(lock, 0, 1)); // lock
          pos = *idx;
          atomicAdd(idx, count);
          __threadfence();
          //    printf("thread %d adds element from index %d to index %d\n",
          //    gidx, *idx, (*idx)+count);
          atomicCAS(lock, 1, 0); // release lock

          for (int i = 0; i < count; i++) {
            d_result[pos + i] = temp_result[i];
            //     printf("%f added to final\n", temp_result[i]);
          }
        }
      },
      idx);

  CHECK_ERROR(cudaGetLastError())
  CHECK_ERROR(cudaDeviceSynchronize())

  // release the memory allocated for the lock
  CHECK_ERROR(cudaFree(lock))
}

template <typename Algorithm, typename R>
void parallel_filter(size_t size, Algorithm algorithm, int *idx,
                     vecmem::data::vector_view<R> &result_view,
                     vecmem::data::vector_view<R> partial_result) {

  parallel_filter(vecpar::cuda::getReduceConfig<R>(size), size, algorithm, idx,
                  result_view, partial_result);
}
} // namespace internal
#endif // VECPAR_CUDA_INTERNAL_HPP
