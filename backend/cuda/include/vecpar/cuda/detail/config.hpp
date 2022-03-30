#ifndef VECPAR_CUDA_CONFIG_HPP
#define VECPAR_CUDA_CONFIG_HPP

#include "vecpar/core/definitions/config.hpp"

namespace vecpar::cuda {

static config getDefaultConfig(size_t size) {
  int nThreadsPerBlock = 256;

  // If the arrays are not even this large, then reduce the value to the
  // size of the arrays.
  if (static_cast<int>(size) < nThreadsPerBlock) {
    nThreadsPerBlock = static_cast<int>(size);
  }
  const int nBlocks =
      static_cast<int>((size + nThreadsPerBlock - 1) / nThreadsPerBlock);
  return config{nBlocks, nThreadsPerBlock, 0};
}

template <typename Ri> static config getReduceConfig(size_t size) {
  int nThreadsPerBlock = 256; // must be power of 2

  if (static_cast<int>(size) < nThreadsPerBlock) {
    nThreadsPerBlock = (size > 64) ? 64 : 32; // less than 32 is useless
  }

  const int nBlocks =
      static_cast<int>((size + nThreadsPerBlock - 1) / nThreadsPerBlock);
  // TODO: here check if the needed shared memory fits the GPU
  return {nBlocks, nThreadsPerBlock, nThreadsPerBlock * sizeof(Ri)};
}
}
#endif //VECPAR_CUDA_CONFIG_HPP
