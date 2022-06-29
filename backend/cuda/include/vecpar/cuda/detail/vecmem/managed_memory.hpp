#ifndef VECPAR_CUDA_MANMEM_HPP
#define VECPAR_CUDA_MANMEM_HPP

#include <vecmem/containers/data/vector_view.hpp>
#include <vecmem/memory/cuda/managed_memory_resource.hpp>

#include "internal.hpp"

namespace vecpar::cuda {

template <
    typename Algorithm, typename R = typename Algorithm::intermediate_result_t, typename T,
    typename... Arguments,
    typename std::enable_if<!std::is_same<T, R>::value, void>::type * = nullptr>
R &parallel_map(Algorithm algorithm,
                vecmem::cuda::managed_memory_resource &mr,
                vecpar::config config,
                T &data,
                Arguments... args) {

  R *map_result = new R(data.size(), &mr);
  auto map_view = vecmem::get_data(*map_result);

  internal::parallel_map(config, data.size(), algorithm, map_view,
                         vecmem::get_data(data), args...);
  return *map_result;
}

template <
    typename Algorithm, typename R = typename Algorithm::intermediate_result_t, typename T,
    typename... Arguments,
    typename std::enable_if<!std::is_same<T, R>::value, void>::type * = nullptr>
R &parallel_map(Algorithm algorithm,
                vecmem::cuda::managed_memory_resource &mr,
                T &data, Arguments... args) {
  return parallel_map(algorithm, mr, cuda::getDefaultConfig(data.size()), data,
                      args...);
}

template <
    typename Algorithm, typename R = typename Algorithm::intermediate_result_t, typename T,
    typename... Arguments,
    typename std::enable_if<std::is_same<T, R>::value, void>::type * = nullptr>
R &parallel_map(Algorithm algorithm,
                __attribute__((unused))
                vecmem::cuda::managed_memory_resource &mr,
                vecpar::config config, T &data,
                Arguments... args) {

  auto map_view = vecmem::get_data(data);
  internal::parallel_map(config, data.size(), algorithm, map_view, args...);
  return data;
}

template <
    typename Algorithm, typename R = typename Algorithm::intermediate_result_t, typename T,
    typename... Arguments,
    typename std::enable_if<std::is_same<T, R>::value, void>::type * = nullptr>
R &parallel_map(Algorithm algorithm,
                                __attribute__((unused))
                                vecmem::cuda::managed_memory_resource &mr,
                                T &data, Arguments... args) {
  return parallel_map(algorithm, mr, cuda::getDefaultConfig(data.size()), data,
                      args...);
}

template <typename Algorithm, typename T>
typename T::value_type &parallel_reduce(Algorithm algorithm,
                   __attribute__((unused))
                   vecmem::cuda::managed_memory_resource &mr,
                   T &data) {

  typename T::value_type *d_result;
  cudaMallocManaged(&d_result, sizeof(typename T::value_type));
  memset(d_result, 0, sizeof(typename T::value_type));

  internal::parallel_reduce(data.size(), algorithm, d_result,
                            vecmem::get_data(data));

  return *d_result;
}

template <typename Algorithm, typename T>
T &parallel_filter(Algorithm algorithm,
                                   vecmem::cuda::managed_memory_resource &mr,
                                   T &data) {
  T* result = new T(data.size(), &mr);
  auto result_view = vecmem::get_data(*result);

  int *idx; // global index
  CHECK_ERROR(cudaMallocManaged((void **)&idx, sizeof(int)))
  *idx = 0;

  internal::parallel_filter(data.size(), algorithm, idx, result_view,
                            vecmem::get_data(data));
  result->resize(*idx);

  // release the memory allocated for the index
  CHECK_ERROR(cudaFree(idx))

  return *result;
}

template <class Algorithm, typename Result, typename R, typename T, typename... Arguments>
Result &parallel_map_reduce(Algorithm algorithm,
                       vecmem::cuda::managed_memory_resource &mr,
                       vecpar::config config, T &data,
                       Arguments... args) {

  return parallel_reduce(algorithm, mr,
                         parallel_map(algorithm, mr, config, data, args...));
}

template <class Algorithm, typename Result, typename R, typename T, typename... Arguments>
Result &parallel_map_reduce(Algorithm algorithm,
                       vecmem::cuda::managed_memory_resource &mr,
                       T &data, Arguments... args) {

  return parallel_map_reduce<Algorithm, Result, R, T, Arguments...>(
      algorithm, mr, cuda::getDefaultConfig(data.size()), data, args...);
}

template <class Algorithm, typename R, typename T, typename... Arguments>
R &parallel_map_filter(
    Algorithm algorithm, vecmem::cuda::managed_memory_resource &mr,
    vecpar::config config, T &data, Arguments... args) {

  return parallel_filter(algorithm, mr,
                         parallel_map(algorithm, mr, config, data, args...));
}

template <class Algorithm, typename R, typename T, typename... Arguments>
R &
parallel_map_filter(Algorithm algorithm,
                    vecmem::cuda::managed_memory_resource &mr,
                    T &data, Arguments... args) {

  return parallel_map_filter<Algorithm, R, T, Arguments...>(
      algorithm, mr, cuda::getDefaultConfig(data.size()), data, args...);
}

} // namespace vecpar::cuda
#endif // VECPAR_CUDA_MANMEM_HPP
