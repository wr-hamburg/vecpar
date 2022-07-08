
#ifndef VECPAR_C_HPP
#define VECPAR_C_HPP

#include <cuda.h>

template <typename Function, typename... Arguments>
__global__ void kernel(size_t size, Function f, Arguments... args) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  f(idx, args...);
}
#endif // VECPAR_C_HPP
