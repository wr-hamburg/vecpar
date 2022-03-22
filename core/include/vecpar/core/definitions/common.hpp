#ifndef VECPAR_COMMON_HPP
#define VECPAR_COMMON_HPP

#ifdef __CUDACC__
#define TARGET __host__ __device__
#else
#define TARGET
#endif

#endif //VECPAR_COMMON_HPP
