#ifndef VECPAR_KERNELS_HPP
#define VECPAR_KERNELS_HPP

#include <cuda.h>

namespace vecpar::cuda {

    template<typename Function, typename... Arguments>
    __global__ void kernel(int size,
                           Function f,
                           Arguments... args) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= size)
            return;
        f(idx, args...);
    }

    template<typename Function, typename... Arguments>
    __global__ void rkernel(int *lock,
                            int size,
                            Function f,
                            Arguments... args) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= size)
            return;
        f(lock, args...);
    }
}
#endif //VECPAR_KERNELS_HPP
