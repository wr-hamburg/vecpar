#ifndef VECPAR_ALG2_CUDA_HPP
#define VECPAR_ALG2_CUDA_HPP

#include <cuda.h>

#include <vecmem/containers/vector.hpp>
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/containers/data/vector_view.hpp>

#include <vecmem/memory/cuda/managed_memory_resource.hpp>

#include "vecpar/core/definitions/config.hpp"
#include "vecpar/cuda/detail/config.hpp"
#include "vecpar/cuda/detail/cuda_utils.hpp"

#include "../../common/data_types.hpp"
#include "../../common/algorithm/algorithm.hpp"

__global__ void kernel_alg_2(vecmem::data::vector_view<int> data_view, X* x, double* d_result) {
    vecmem::device_vector<int> d_data(data_view);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_data.size())
        return;
    atomicAdd(d_result, d_data[idx] * x->f());
}

class test_algorithm_2_cuda :
        public traccc::algorithm<double(vecmem::vector<int>&, X)>{

public:
    test_algorithm_2_cuda(vecmem::memory_resource& mr): algorithm(), m_mr(mr) {}

    double operator() (vecmem::vector<int>& data, X more_data) override {
        double* result = (double*) malloc(sizeof(double));

        double* d_result;
        CHECK_ERROR(cudaMalloc((void**)&d_result, sizeof(double)));
        CHECK_ERROR(cudaMemset(d_result, 0, sizeof(double)));

        X* d_extra;
        CHECK_ERROR(cudaMalloc((void**)&d_extra, sizeof(X)));
        CHECK_ERROR(cudaMemcpy(d_extra, &more_data, sizeof(X), cudaMemcpyHostToDevice));

        vecpar::config c = vecpar::cuda::getDefaultConfig(data.size());
        auto view = vecmem::get_data(data);
        kernel_alg_2<<<c.m_gridSize, c.m_blockSize>>>(view, d_extra, d_result);
        CHECK_ERROR(cudaGetLastError());
        CHECK_ERROR(cudaDeviceSynchronize());

        CHECK_ERROR(cudaMemcpy(result, d_result, sizeof(double), cudaMemcpyDeviceToHost));
        return *result;
    }

private:
    vecmem::memory_resource& m_mr;
};

#endif