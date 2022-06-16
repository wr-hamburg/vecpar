#ifndef VECPAR_KERNELS_HPP
#define VECPAR_KERNELS_HPP

#include <cuda.h>

namespace vecpar::cuda {

template <typename Function, typename... Arguments>
__global__ void kernel(size_t size, Function f, Arguments... args) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  f(idx, args...);
}

template <typename Function, typename... Arguments>
__global__ void rkernel(int *lock, size_t size, Function f, Arguments... args) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  f(lock, args...);
}
} // namespace vecpar::cuda
#endif // VECPAR_KERNELS_HPP
