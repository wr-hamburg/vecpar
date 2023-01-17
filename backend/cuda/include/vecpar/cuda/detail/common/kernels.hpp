#ifndef VECPAR_KERNELS_HPP
#define VECPAR_KERNELS_HPP

#include <cuda.h>

namespace vecpar::cuda {

// uncomment grid_constants when clang will support cuda 11.7; they are ignored until then
template <typename Function, typename... Arguments>
__global__ void kernel(const/* __grid_constant__*/ size_t size, const /*__grid_constant__ */ Function f, Arguments... args) {
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
