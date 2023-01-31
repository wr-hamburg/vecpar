#ifndef VECPAR_COMMON_HPP
#define VECPAR_COMMON_HPP

#ifdef __CUDA__
#define TARGET __host__ __device__ __forceinline__
#else
#define TARGET inline
#endif

#ifndef NDEBUG
#define DBG 0
#else
#define DBG 0
#endif

#define DEBUG_ACTION(action)                                                   \
  if (DBG) {                                                                   \
    action                                                                     \
  }

#endif // VECPAR_COMMON_HPP
