#ifndef VECPAR_ALG2_CUDA_MM_HPP
#define VECPAR_ALG2_CUDA_MM_HPP

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

class test_algorithm_2_cuda_mm
    : public traccc::algorithm<double(vecmem::vector<int> &, X)> {

public:
  test_algorithm_2_cuda_mm(vecmem::memory_resource &mr)
      : algorithm(), m_mr(mr) {}

  double operator()(vecmem::vector<int> &data, X more_data) override {

    vecpar::config c = vecpar::cuda::getDefaultConfig(data.size());

    // data from input collection
    auto view = vecmem::get_data(data);

    // allocate space for result
    double *d_result;
    CHECK_ERROR(cudaMallocManaged((void **)&d_result, sizeof(double)));
    CHECK_ERROR(cudaMemset(d_result, 0, sizeof(double)));

    // call kernel
    kernel<<<c.m_gridSize, c.m_blockSize, c.m_memorySize>>>(
        data.size(),
        [=] __device__(int idx, X) mutable {
          vecmem::device_vector<int> dv_data(view);
          atomicAdd(d_result, dv_data[idx] * more_data.f());
        },
        more_data);

    CHECK_ERROR(cudaGetLastError())
    CHECK_ERROR(cudaDeviceSynchronize())

    return *d_result;
  }

private:
  vecmem::memory_resource &m_mr;
};

#endif