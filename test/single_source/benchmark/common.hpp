
#ifndef VECPAR_C_HPP
#define VECPAR_C_HPP

#include <cuda.h>

__global__ void kernel(vecmem::data::vector_view<float> x_view,
                       vecmem::data::vector_view<float> y_view, float d_a) {
  vecmem::device_vector<float> d_x(x_view);
  vecmem::device_vector<float> d_y(y_view);
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= d_x.size())
    return;
  d_y[idx] = d_x[idx] * d_a + d_y[idx];
}

#endif // VECPAR_C_HPP
