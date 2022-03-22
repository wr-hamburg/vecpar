#ifndef VECPAR_CUDA_INTERNAL_HPP
#define VECPAR_CUDA_INTERNAL_HPP

#include <vecmem/containers/device_vector.hpp>
#include <vecmem/containers/data/vector_view.hpp>

#include "vecpar/core/definitions/config.hpp"

#include "cuda_utils.hpp"

namespace internal {
//       typename std::enable_if<!std::is_same<T, R>::value, void>::type* = nullptr

    template<typename Algorithm,
            typename R,
            typename T,
            typename... Arguments>
    void parallel_map(vecpar::config c,
                      int size,
                      Algorithm algorithm,
                      vecmem::data::vector_view<R> &result,
                      vecmem::data::vector_view<T> data,
                      Arguments... args) {

        vecpar::cuda::kernel<<<c.gridSize, c.blockSize, c.memorySize>>>(size,
                [=] __device__ (int idx, Arguments... a) mutable {
            vecmem::device_vector<T> dv_data(data);
            vecmem::device_vector<R> dv_result(result);
       //     printf("[mapper] data[%d]=%f\n", idx, dv_data[idx]);
            algorithm.map(dv_result[idx], dv_data[idx], a...);
         //   printf("[mapper] result[%d]=%f\n", idx, dv_result[idx]);
        }, args...);

        CHECK_ERROR(cudaGetLastError());
        CHECK_ERROR(cudaDeviceSynchronize());
    }

    template<typename Algorithm,
            typename TT,
            typename... Arguments>
    void parallel_map(vecpar::config c,
                      int size,
                      Algorithm algorithm,
                      vecmem::data::vector_view<TT> &input_output,
                      Arguments... args) {

        vecpar::cuda::kernel<<<c.gridSize, c.blockSize, c.memorySize>>>(size,
                [=] __device__ (int idx, Arguments... a) mutable {
            vecmem::device_vector<TT> dv_data(input_output);
            //     printf("[mapper] data[%d]=%f\n", idx, dv_data[idx]);
            algorithm.map(dv_data[idx], a...);
            //   printf("[mapper] result[%d]=%f\n", idx, dv_result[idx]);
        }, args...);

        CHECK_ERROR(cudaGetLastError());
        CHECK_ERROR(cudaDeviceSynchronize());
    }

    template<typename Algorithm, typename R, typename T, typename... Arguments>
    void parallel_map(int size,
                      Algorithm algorithm,
                      vecmem::data::vector_view<R> &result,
                      vecmem::data::vector_view<T> data ,
                      Arguments... args) {

        vecpar::config c = vecpar::getDefaultConfig(size);
        //printf("nBlocks:%d, nThreads:%d, memorySize:%zu\n", c.gridSize, c.blockSize, c.memorySize);
        parallel_map(c, size, algorithm, result, data, args...);
    }

    template<typename Algorithm, typename TT, typename... Arguments>
    void parallel_map(int size,
                      Algorithm algorithm,
                      vecmem::data::vector_view<TT> &input_output,
                      Arguments... args) {

        vecpar::config c = vecpar::getDefaultConfig(size);
        //printf("nBlocks:%d, nThreads:%d, memorySize:%zu\n", c.gridSize, c.blockSize, c.memorySize);
        parallel_map(c, size, algorithm, input_output, args...);
    }

    /// based on https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    /// more efficient versions using thread-block parametrization can be implemented
    template<typename Algorithm, typename R>
    void parallel_reduce(vecpar::config c,
                         int size,
                         Algorithm algorithm,
                         R *result,
                         vecmem::data::vector_view<R> partial_result_view) {

        int* lock;//mutex.
        cudaMallocManaged((void**)&lock, sizeof(int));
        *lock = 0;

        vecpar::cuda::rkernel<<<c.gridSize, c.blockSize, c.memorySize>>>(lock, size,
                [=] __device__ (int* lock) mutable {
            vecmem::device_vector <R> partial_result(partial_result_view);
            extern __shared__ R temp[];

            int tid = threadIdx.x;
            temp[tid] = partial_result[tid + blockIdx.x * blockDim.x];
            int gidx = threadIdx.x + blockIdx.x * blockDim.x;
            //         printf("gidx %d starts with %f\n", gidx, temp[tid]);

            for (int d = blockDim.x >> 1; d >= 1; d >>= 1) {
                __syncthreads();

                /// for odd size and (larger) even number of threads, make sure
                /// we do not read from outside of the array
                bool within_array = ((tid + d) < blockDim.x) && (gidx + d < size);
                //printf("tid = %d, d = %d, gidx = %d, within_array? = %d (%d)\n", tid, d, gidx,within_array, tid+d);

                if (tid < d && within_array) {
                    //    printf("thread *%d*: read from index %d, %f + %f\n", gidx, tid+d, temp[tid], temp[tid+d]);
                    algorithm.reduce(&temp[tid], temp[tid + d]);
                }
            }

            if (tid == 0) {
                do {} while (atomicCAS(lock, 0, 1)); // lock
                //     printf("thread %d: %f + %f \n", gidx, *result, temp[0]);
                algorithm.reduce(result, temp[0]);
                __threadfence(); // wait for write completion
                atomicCAS(lock, 1, 0); // release lock
            }

        });

        CHECK_ERROR(cudaGetLastError());
        CHECK_ERROR(cudaDeviceSynchronize());
    }

    template<typename Algorithm, typename R>
    void parallel_reduce(int size,
                         Algorithm algorithm,
                         R *result,
                         vecmem::data::vector_view<R> partial_result) {
        vecpar::config c = vecpar::getReduceConfig<R>(size);
        //     printf("nBlocks:%d, nThreads:%d, memorySize:%zu\n", c.gridSize, c.blockSize, c.memorySize);
        parallel_reduce(c, size, algorithm, result, partial_result);
    }

    template<typename Algorithm, typename R>
    void parallel_filter(vecpar::config c,
                         int size,
                         Algorithm algorithm,
                         int* idx,
                         vecmem::data::vector_view<R>& result_view,
                         vecmem::data::vector_view<R> partial_result_view) {

        int* lock;//mutex.
        cudaMallocManaged((void**)&lock, sizeof(int));
        *lock = 0;

        vecpar::cuda::rkernel<<<c.gridSize, c.blockSize, c.memorySize>>>(lock, size,
                [=] __device__  (int* lock, int* idx) mutable {

            vecmem::device_vector<R> d_result(result_view);
            vecmem::device_vector<R> partial_result(partial_result_view);
            extern __shared__ R temp[];

            int gidx = threadIdx.x + blockIdx.x * blockDim.x;
            if (gidx > size)
                return;

            int tid = threadIdx.x;
            temp[tid] = partial_result[tid + blockIdx.x * blockDim.x];
            //    printf("thread %d loads element %f\n", gidx, temp[tid]);
            __syncthreads();

            if (tid == 0) {
                R temp_result[256];// TODO: this should not be hardcoded;
                int count = 0;
                //   printf("blocks = %d, threads = %d\n", c.gridSize, c.blockSize);
                for (int i = 0; i < c.blockSize && tid + i < blockDim.x && gidx + i < size ; i++) {
                    if (algorithm.filter(temp[tid+i])) {
                        temp_result[count] = temp[tid+i];
                        count++;
                        //printf("%f added to temp\n", temp[tid+i]);
                    }
                }

                int pos = 0; /// pos where to add in the global result

                do {} while (atomicCAS(lock, 0, 1)); // lock
                pos = *idx;
                atomicAdd(idx, count);
                atomicCAS(lock, 1, 0); // release lock

                //printf("element %f fits the list\n", temp[tid]);
                for (int i = 0; i < count; i++) {
                    d_result[pos + i] = temp_result[i];
                    // printf("%f added to final\n", temp_result[i]);
                }
            }
        }, idx);

        CHECK_ERROR(cudaGetLastError());
        CHECK_ERROR(cudaDeviceSynchronize());
    }

    template<typename Algorithm, typename R>
    void parallel_filter(int size,
                         Algorithm algorithm,
                         int* idx,
                         vecmem::data::vector_view<R>& result_view,
                         vecmem::data::vector_view<R> partial_result) {
        vecpar::config c = vecpar::getReduceConfig<R>(size);
        parallel_filter(c, size, algorithm, idx, result_view, partial_result);
    }

}
#endif //VECPAR_CUDA_INTERNAL_HPP
