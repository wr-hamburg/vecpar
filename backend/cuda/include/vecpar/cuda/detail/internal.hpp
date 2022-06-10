#ifndef VECPAR_CUDA_INTERNAL_HPP
#define VECPAR_CUDA_INTERNAL_HPP

#include <vecmem/containers/device_vector.hpp>
#include <vecmem/containers/data/vector_view.hpp>

#include "vecpar/core/definitions/config.hpp"
#include "vecpar/core/definitions/common.hpp"
#include "vecpar/cuda/detail/cuda_utils.hpp"
#include "vecpar/cuda/detail/config.hpp"
#include "vecpar/cuda/detail/kernels.hpp"


namespace internal {

    template<typename Algorithm, typename R, typename T, typename... Arguments>
    void parallel_map(vecpar::config c, size_t size, Algorithm algorithm,
                      vecmem::data::vector_view<R> &result,
                      vecmem::data::vector_view<T> data, Arguments... args) {

        // make sure that an empty config ends up to be used
        if (c.isEmpty()) {
            c = vecpar::cuda::getDefaultConfig(size);
        }

        DEBUG_ACTION(printf("[MAP] nBlocks:%d, nThreads:%d, memorySize:%zu\n",
                            c.m_gridSize, c.m_blockSize, c.m_memorySize);)

        vecpar::cuda::kernel<<<c.m_gridSize, c.m_blockSize, c.m_memorySize>>>(size,
                        [=] __device__(int idx, Arguments... a) mutable {
            vecmem::device_vector<T> dv_data(data);
            vecmem::device_vector<R> dv_result(result);
            //     printf("[mapper] data[%d]=%f\n", idx, dv_data[idx]);
            algorithm.map(dv_result[idx], dv_data[idx], a...);
            //   printf("[mapper] result[%d]=%f\n", idx, dv_result[idx]);
        },
        args...);

        CHECK_ERROR(cudaGetLastError())
        CHECK_ERROR(cudaDeviceSynchronize())
    }

    template<typename Algorithm, typename TT, typename... Arguments>
    void parallel_map(vecpar::config c, size_t size, Algorithm algorithm,
                      vecmem::data::vector_view<TT> &input_output,
                      Arguments... args) {

        // make sure that an empty config ends up to be used
        if (c.isEmpty()) {
            c = vecpar::cuda::getDefaultConfig(size);
        }

        DEBUG_ACTION(printf("[MAP] nBlocks:%d, nThreads:%d, memorySize:%zu\n",
                            c.m_gridSize, c.m_blockSize, c.m_memorySize);)

        vecpar::cuda::kernel<<<c.m_gridSize, c.m_blockSize, c.m_memorySize>>>(size,
                        [=] __device__(int idx, Arguments... a) mutable {
            vecmem::device_vector<TT> dv_data(input_output);
            //     printf("[mapper] data[%d]=%f\n", idx, dv_data[idx]);
            algorithm.map(dv_data[idx], a...);
            //     printf("[mapper] result[%d]=%f\n", idx, dv_data[idx]);
        },
        args...);

        CHECK_ERROR(cudaGetLastError())
        CHECK_ERROR(cudaDeviceSynchronize())
    }

    template<typename Algorithm, typename R, typename T, typename... Arguments>
    void parallel_map(size_t size, Algorithm algorithm,
                      vecmem::data::vector_view<R> &result,
                      vecmem::data::vector_view<T> data, Arguments... args) {

        parallel_map(vecpar::cuda::getDefaultConfig(size),
                     size, algorithm, result, data, args...);
    }

    template<typename Algorithm, typename TT, typename... Arguments>
    void parallel_map(size_t size, Algorithm algorithm,
                      vecmem::data::vector_view<TT> &input_output,
                      Arguments... args) {

        parallel_map(vecpar::cuda::getDefaultConfig(size),
                     size, algorithm, input_output, args...);
    }

    /// based on https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    /// more efficient versions using thread-block parametrization can be implemented
    template<typename Algorithm, typename R>
    void parallel_reduce(vecpar::config c, size_t size, Algorithm algorithm,
                         R *result,
                         vecmem::data::vector_view<R> partial_result_view) {

        int *lock; // mutex.
        cudaMallocManaged((void **) &lock, sizeof(int));
        *lock = 0;

        // make sure that an empty config ends up to be used
        if (c.isEmpty()) {
            c = vecpar::cuda::getReduceConfig<R>(size);
        }

        DEBUG_ACTION(printf("[REDUCE] nBlocks:%d, nThreads:%d, memorySize:%zu\n",
                            c.m_gridSize, c.m_blockSize, c.m_memorySize);)

        vecpar::cuda::rkernel<<<c.m_gridSize, c.m_blockSize, c.m_memorySize>>>(
                lock, size, [=] __device__(int * lock) mutable {
            vecmem::device_vector<R> partial_result(partial_result_view);
            extern __shared__ R
            temp[];

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
    }

    template<typename Algorithm, typename R>
    void parallel_reduce(size_t size, Algorithm algorithm, R *result,
                         vecmem::data::vector_view<R> partial_result) {

        parallel_reduce(vecpar::cuda::getReduceConfig<R>(size),
                        size, algorithm, result, partial_result);
    }

    template<typename Algorithm, typename R>
    void parallel_filter(vecpar::config c, size_t size, Algorithm algorithm,
                         int *idx, vecmem::data::vector_view<R> &result_view,
                         vecmem::data::vector_view<R> partial_result_view) {

        int *lock; // mutex.
        cudaMallocManaged((void **) &lock, sizeof(int));
        *lock = 0;

        // make sure that an empty config ends up to be used
        if (c.isEmpty()) {
            c = vecpar::cuda::getReduceConfig<R>(size);
        }

        DEBUG_ACTION(printf("[FILTER] nBlocks:%d, nThreads:%d, memorySize:%zu\n",
                            c.m_gridSize, c.m_blockSize, c.m_memorySize);)

        vecpar::cuda::rkernel<<<c.m_gridSize, c.m_blockSize, c.m_memorySize>>>(
                lock, size, [=] __device__(int * lock, int * idx) mutable {
            vecmem::device_vector<R> d_result(result_view);
            vecmem::device_vector<R> partial_result(partial_result_view);
            extern __shared__ R
            temp[];

            size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
            if (gidx > size)
                return;

            size_t tid = threadIdx.x;
            temp[tid] = partial_result[tid + blockIdx.x * blockDim.x];
            //    printf("thread %d loads element %f\n", gidx, temp[tid]);
            __syncthreads();

            if (tid == 0) {
                //    printf("First thread from block: %d\n ", blockIdx.x);

                R temp_result[256]; // TODO: this should not be hardcoded;
                int count = 0;
                //printf("blocks = %d, threads = %d\n", c.gridSize, c.blockSize);
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
                //    printf("thread %d adds element from index %d to index %d\n", gidx, *idx, (*idx)+count);
                atomicCAS(lock, 1, 0); // release lock

                // printf("element %f fits the list\n", temp[tid]);
                //    printf("Has to copy %u elements in the final vector \n", count);
                for (int i = 0; i < count; i++) {
                    d_result[pos + i] = temp_result[i];
                    //     printf("%f added to final\n", temp_result[i]);
                }
            }
        }, idx);

        CHECK_ERROR(cudaGetLastError())
        CHECK_ERROR(cudaDeviceSynchronize())
    }

    template<typename Algorithm, typename R>
    void parallel_filter(size_t size, Algorithm algorithm, int *idx,
                         vecmem::data::vector_view<R> &result_view,
                         vecmem::data::vector_view<R> partial_result) {

        parallel_filter(vecpar::cuda::getReduceConfig<R>(size),
                        size, algorithm, idx, result_view, partial_result);
    }
}
#endif //VECPAR_CUDA_INTERNAL_HPP
