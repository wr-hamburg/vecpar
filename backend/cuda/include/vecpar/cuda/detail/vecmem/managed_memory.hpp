#ifndef VECPAR_CUDA_MANMEM_HPP
#define VECPAR_CUDA_MANMEM_HPP

#include <vecmem/containers/data/vector_view.hpp>
#include <vecmem/memory/cuda/managed_memory_resource.hpp>

#include "internal.hpp"

namespace vecpar::cuda {

/// map with config
template <typename Algorithm,
          typename R = typename Algorithm::intermediate_result_t, typename T,
          typename... Arguments>
requires vecpar::detail::is_map<Algorithm, R, T, Arguments...> R &
parallel_map(Algorithm algorithm, vecmem::cuda::managed_memory_resource &mr,
             vecpar::config config, T &in_1, Arguments &...args) {

  R *map_result = new R(in_1.size(), &mr);
  auto map_view = vecmem::get_data(*map_result);

  auto input = get_view_or_obj(in_1, args...);

  auto fn = [&]<typename... P>(P & ...params) {
    return internal::parallel_map<Algorithm, R, T, Arguments...>(
        config, in_1.size(), algorithm, map_view, params...);
  };

  std::apply(fn, input);

  return *map_result;
}

/// map without config
template <typename Algorithm,
          typename R = typename Algorithm::intermediate_result_t, typename T,
          typename... Arguments>
requires vecpar::detail::is_map<Algorithm, R, T, Arguments...> R &
parallel_map(Algorithm algorithm, vecmem::cuda::managed_memory_resource &mr,
             T &data, Arguments &...args) {
  return parallel_map(algorithm, mr, cuda::getDefaultConfig(data.size()), data,
                      args...);
}
/// mmap with config
template <typename Algorithm,
          typename R = typename Algorithm::intermediate_result_t, typename T,
          typename... Arguments>
requires vecpar::detail::is_mmap<Algorithm, T, Arguments...> R &
parallel_map(Algorithm algorithm,
             __attribute__((unused)) vecmem::cuda::managed_memory_resource &mr,
             vecpar::config config, T &in_out_1, Arguments &...args) {

  auto input = get_view_or_obj(in_out_1, args...);

  auto fn = [&]<typename... P>(P & ...params) {
    return internal::parallel_mmap<Algorithm, T, Arguments...>(
        config, in_out_1.size(), algorithm, params...);
  };

  std::apply(fn, input);

  return in_out_1;
}

/// mmap without config
template <typename Algorithm,
          typename R = typename Algorithm::intermediate_result_t, typename T,
          typename... Arguments>
requires vecpar::detail::is_mmap<Algorithm, R, Arguments...> R &
parallel_map(Algorithm algorithm,
             __attribute__((unused)) vecmem::cuda::managed_memory_resource &mr,
             T &data, Arguments &...args) {
  return parallel_map(algorithm, mr, cuda::getDefaultConfig(data.size()), data,
                      args...);
}

template <typename Algorithm, typename T>
typename T::value_type &
parallel_reduce(Algorithm algorithm,
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
                   vecmem::cuda::managed_memory_resource &mr, T &data) {
  T *result = new T(data.size(), &mr);
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

template <class Algorithm, typename Result, typename R, typename T,
          typename... Arguments>
Result &parallel_map_reduce(Algorithm algorithm,
                            vecmem::cuda::managed_memory_resource &mr,
                            vecpar::config config, T &data,
                            Arguments &...args) {

  return parallel_reduce(algorithm, mr,
                         parallel_map<Algorithm, R, T, Arguments...>(
                             algorithm, mr, config, data, args...));
}

template <class Algorithm, typename Result, typename R, typename T,
          typename... Arguments>
Result &parallel_map_reduce(Algorithm algorithm,
                            vecmem::cuda::managed_memory_resource &mr, T &data,
                            Arguments &...args) {

  return parallel_map_reduce<Algorithm, Result, R, T, Arguments...>(
      algorithm, mr, cuda::getDefaultConfig(data.size()), data, args...);
}

template <class Algorithm, typename R, typename T, typename... Arguments>
R &parallel_map_filter(Algorithm algorithm,
                       vecmem::cuda::managed_memory_resource &mr,
                       vecpar::config config, T &data, Arguments &...args) {

  return parallel_filter(algorithm, mr,
                         parallel_map<Algorithm, R, T, Arguments...>(
                             algorithm, mr, config, data, args...));
}

template <class Algorithm, typename R, typename T, typename... Arguments>
R &parallel_map_filter(Algorithm algorithm,
                       vecmem::cuda::managed_memory_resource &mr, T &data,
                       Arguments &...args) {

  return parallel_map_filter<Algorithm, R, T, Arguments...>(
      algorithm, mr, cuda::getDefaultConfig(data.size()), data, args...);
}

} // namespace vecpar::cuda
#endif // VECPAR_CUDA_MANMEM_HPP
