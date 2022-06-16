#ifndef VECPAR_CUDA_UTILS_HPP
#define VECPAR_CUDA_UTILS_HPP

#include <assert.h>
#include <cuda.h>

#define CHECK_ERROR(ans)                                                       \
  { cudaCheck((ans), __FILE__, __LINE__); }

inline void cudaCheck(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDAcheck: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}
#endif // VECPAR_CUDA_UTILS_HPP
