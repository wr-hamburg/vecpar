#ifndef VECPAR_KERNELS_HPP
#define VECPAR_KERNELS_HPP

#include <cuda.h>

namespace vecpar::cuda {

template <typename Function, typename... Arguments>
__global__ void kernel(const size_t size, const Function f, Arguments... args) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  f(idx, args...);
}

template <typename Function, typename... Arguments>
__global__ void rkernel(int *lock, const size_t size,
                        const /*__grid_constant__*/ Function f,
                        Arguments... args) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  f(lock, args...);
}
} // namespace vecpar::cuda
#endif // VECPAR_KERNELS_HPP
