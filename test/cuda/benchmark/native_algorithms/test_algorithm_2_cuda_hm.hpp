#ifndef VECPAR_ALG2_CUDA_HPP
#define VECPAR_ALG2_CUDA_HPP

#include <cuda.h>

#include <vecmem/containers/data/vector_view.hpp>
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/containers/vector.hpp>

#include <vecmem/memory/cuda/managed_memory_resource.hpp>

#include "vecpar/core/definitions/config.hpp"
#include "vecpar/cuda/detail/common/config.hpp"
#include "vecpar/cuda/detail/common/cuda_utils.hpp"

#include "../../../common/algorithm/algorithm.hpp"
#include "../../../common/data_types.hpp"

#include "common.hpp"

vecmem::cuda::device_memory_resource d_mem;
vecmem::cuda::copy copy;

class test_algorithm_2_cuda_hm
    : public traccc::algorithm<double(vecmem::vector<int> &, X)> {

public:
  test_algorithm_2_cuda_hm(vecmem::memory_resource &mr)
      : algorithm(), m_mr(mr) {}

  double operator()(vecmem::vector<int> &data, X more_data) override {
    vecpar::config c = vecpar::cuda::getDefaultConfig(data.size());

    double *result = (double *)malloc(sizeof(double));

    double *d_result;
    CHECK_ERROR(cudaMalloc((void **)&d_result, sizeof(double)));
    CHECK_ERROR(cudaMemset(d_result, 0, sizeof(double)));

    X *d_extra;
    CHECK_ERROR(cudaMalloc((void **)&d_extra, sizeof(X)));
    CHECK_ERROR(
        cudaMemcpy(d_extra, &more_data, sizeof(X), cudaMemcpyHostToDevice));

    auto buffer = copy.to(vecmem::get_data(data), d_mem,
                          vecmem::copy::type::host_to_device);
    auto view = vecmem::get_data(buffer);

    // call kernel
    kernel<<<c.m_gridSize, c.m_blockSize, c.m_memorySize>>>(
        data.size(),
        [=] __device__(int idx, X *) mutable {
          vecmem::device_vector<int> dv_data(view);
          atomicAdd(d_result, dv_data[idx] * d_extra->f());
        },
        d_extra);

    CHECK_ERROR(cudaGetLastError());
    CHECK_ERROR(cudaDeviceSynchronize());

    CHECK_ERROR(
        cudaMemcpy(result, d_result, sizeof(double), cudaMemcpyDeviceToHost));
    return *result;
  }

private:
  vecmem::memory_resource &m_mr;
};

#endif