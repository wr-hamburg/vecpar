#ifndef VECPAR_CUDA_MANMEM_HPP
#define VECPAR_CUDA_MANMEM_HPP

#include <vecmem/containers/data/vector_view.hpp>
#include <vecmem/memory/cuda/managed_memory_resource.hpp>

#include "internal.hpp"

namespace vecpar::cuda {
template <typename Algorithm,
          typename R = typename Algorithm::intermediate_result_t, typename T,
          typename... Arguments>
requires vecpar::detail::is_map_1<Algorithm, R, T, Arguments...> R &
parallel_map(Algorithm algorithm, vecmem::cuda::managed_memory_resource &mr,
             vecpar::config config, T &in_1, Arguments &...args) {

  R *map_result = new R(in_1.size(), &mr);
  auto map_view = vecmem::get_data(*map_result);
  auto in_1_view = vecmem::get_data(in_1);

  internal::parallel_map_one<Algorithm, R, T, Arguments...>(
      config, in_1.size(), algorithm, map_view, in_1_view, args...);
  return *map_result;
}

template <typename Algorithm,
          typename R = typename Algorithm::intermediate_result_t, typename T1,
          typename T2, typename... Arguments>
requires vecpar::detail::is_map_2<Algorithm, R, T1, T2, Arguments...> R &
parallel_map(Algorithm algorithm, vecmem::cuda::managed_memory_resource &mr,
             vecpar::config config, T1 &in_1, T2 &in_2, Arguments &...args) {

  R *map_result = new R(in_1.size(), &mr);
  auto map_view = vecmem::get_data(*map_result);
  auto in_1_view = vecmem::get_data(in_1);
  auto in_2_view = vecmem::get_data(in_2);

  internal::parallel_map_two<Algorithm, R, T1, T2, Arguments...>(
      config, in_1.size(), algorithm, map_view, in_1_view, in_2_view, args...);
  return *map_result;
}

template <typename Algorithm,
          typename R = typename Algorithm::intermediate_result_t, typename T1,
          typename T2, typename T3, typename... Arguments>
requires vecpar::detail::is_map_3<Algorithm, R, T1, T2, T3, Arguments...> R &
parallel_map(Algorithm algorithm, vecmem::cuda::managed_memory_resource &mr,
             vecpar::config config, T1 &in_1, T2 &in_2, T3 &in_3,
             Arguments &...args) {

  R *map_result = new R(in_1.size(), &mr);
  auto map_view = vecmem::get_data(*map_result);
  auto in_1_view = vecmem::get_data(in_1);
  auto in_2_view = vecmem::get_data(in_2);
  auto in_3_view = vecmem::get_data(in_3);

  internal::parallel_map_three<Algorithm, R, T1, T2, T3, Arguments...>(
      config, in_1.size(), algorithm, map_view, in_1_view, in_2_view, in_3_view,
      args...);
  return *map_result;
}

template <typename Algorithm,
          typename R = typename Algorithm::intermediate_result_t, typename T1,
          typename T2, typename T3, typename T4, typename... Arguments>
requires vecpar::detail::is_map_4<Algorithm, R, T1, T2, T3, T4, Arguments...>
    R &parallel_map(Algorithm algorithm,
                    vecmem::cuda::managed_memory_resource &mr,
                    vecpar::config config, T1 &in_1, T2 &in_2, T3 &in_3,
                    T4 &in_4, Arguments &...args) {

  R *map_result = new R(in_1.size(), &mr);
  auto map_view = vecmem::get_data(*map_result);
  auto in_1_view = vecmem::get_data(in_1);
  auto in_2_view = vecmem::get_data(in_2);
  auto in_3_view = vecmem::get_data(in_3);
  auto in_4_view = vecmem::get_data(in_4);

  internal::parallel_map_four<Algorithm, R, T1, T2, T4, Arguments...>(
      config, in_1.size(), algorithm, map_view, in_1_view, in_2_view, in_3_view,
      in_4_view, args...);
  return *map_result;
}

template <typename Algorithm,
          typename R = typename Algorithm::intermediate_result_t, typename T1,
          typename T2, typename T3, typename T4, typename T5,
          typename... Arguments>
requires vecpar::detail::is_map_5<Algorithm, R, T1, T2, T3, T4, T5,
                                  Arguments...>
    R &parallel_map(Algorithm algorithm,
                    vecmem::cuda::managed_memory_resource &mr,
                    vecpar::config config, T1 &in_1, T2 &in_2, T3 &in_3,
                    T4 &in_4, T5 &in_5, Arguments &...args) {

  R *map_result = new R(in_1.size(), &mr);
  auto map_view = vecmem::get_data(*map_result);
  auto in_1_view = vecmem::get_data(in_1);
  auto in_2_view = vecmem::get_data(in_2);
  auto in_3_view = vecmem::get_data(in_3);
  auto in_4_view = vecmem::get_data(in_4);
  auto in_5_view = vecmem::get_data(in_5);

  internal::parallel_map_five<Algorithm, R, T1, T2, T3, T4, T5, Arguments...>(
      config, in_1.size(), algorithm, map_view, in_1_view, in_2_view, in_3_view,
      in_4_view, in_5_view, args...);
  return *map_result;
}

template <typename Algorithm,
          typename R = typename Algorithm::intermediate_result_t, typename T,
          typename... Arguments>
requires vecpar::detail::is_map<Algorithm, R, T, Arguments...> R &
parallel_map(Algorithm algorithm, vecmem::cuda::managed_memory_resource &mr,
             T &data, Arguments &...args) {
  return parallel_map(algorithm, mr, cuda::getDefaultConfig(data.size()), data,
                      args...);
}

/// mmap functions 1 to 5 input collections :
template <typename Algorithm,
          typename R = typename Algorithm::intermediate_result_t, typename T1,
          typename... Arguments>
requires vecpar::detail::is_mmap_1<Algorithm, T1, Arguments...> R &
parallel_map(Algorithm algorithm,
             __attribute__((unused)) vecmem::cuda::managed_memory_resource &mr,
             vecpar::config config, T1 &in_out_1, Arguments &...args) {

  auto in_out_1_view = vecmem::get_data(in_out_1);

  internal::parallel_mmap_one<Algorithm, T1, Arguments...>(
      config, in_out_1.size(), algorithm, in_out_1_view, args...);
  return in_out_1;
}

template <typename Algorithm,
          typename R = typename Algorithm::intermediate_result_t, typename T1,
          typename T2, typename... Arguments>
requires vecpar::detail::is_mmap_2<Algorithm, T1, T2, Arguments...> R &
parallel_map(Algorithm algorithm,
             __attribute__((unused)) vecmem::cuda::managed_memory_resource &mr,
             vecpar::config config, T1 &in_out_1, T2 &in_2,
             Arguments &...args) {

  auto in_out_1_view = vecmem::get_data(in_out_1);
  auto in_2_view = vecmem::get_data(in_2);

  internal::parallel_mmap_two<Algorithm, T1, T2, Arguments...>(
      config, in_out_1.size(), algorithm, in_out_1_view, in_2_view, args...);
  return in_out_1;
}

template <typename Algorithm,
          typename R = typename Algorithm::intermediate_result_t, typename T1,
          typename T2, typename T3, typename... Arguments>
requires vecpar::detail::is_mmap_3<Algorithm, T1, T2, T3, Arguments...> R &
parallel_map(Algorithm algorithm,
             __attribute__((unused)) vecmem::cuda::managed_memory_resource &mr,
             vecpar::config config, T1 &in_out_1, T2 &in_2, T3 &in_3,
             Arguments &...args) {

  auto in_out_1_view = vecmem::get_data(in_out_1);
  auto in_2_view = vecmem::get_data(in_2);
  auto in_3_view = vecmem::get_data(in_3);

  internal::parallel_mmap_three<Algorithm, T1, T2, T3, Arguments...>(
      config, in_out_1.size(), algorithm, in_out_1_view, in_2_view, in_3_view,
      args...);
  return in_out_1;
}

template <typename Algorithm,
          typename R = typename Algorithm::intermediate_result_t, typename T1,
          typename T2, typename T3, typename T4, typename... Arguments>
requires vecpar::detail::is_mmap_4<Algorithm, T1, T2, T3, T4, Arguments...> R &
parallel_map(Algorithm algorithm,
             __attribute__((unused)) vecmem::cuda::managed_memory_resource &mr,
             vecpar::config config, T1 &in_out_1, T2 &in_2, T3 &in_3, T4 &in_4,
             Arguments &...args) {

  auto in_out_1_view = vecmem::get_data(in_out_1);
  auto in_2_view = vecmem::get_data(in_2);
  auto in_3_view = vecmem::get_data(in_3);
  auto in_4_view = vecmem::get_data(in_4);

  internal::parallel_mmap_four<Algorithm, T1, T2, T3, T4, Arguments...>(
      config, in_out_1.size(), algorithm, in_out_1_view, in_2_view, in_3_view,
      in_4_view, args...);
  return in_out_1;
}

template <typename Algorithm,
          typename R = typename Algorithm::intermediate_result_t, typename T1,
          typename T2, typename T3, typename T4, typename T5,
          typename... Arguments>
requires vecpar::detail::is_mmap_5<Algorithm, T1, T2, T3, T4, T5, Arguments...>
    R &parallel_map(Algorithm algorithm,
                    __attribute__((unused))
                    vecmem::cuda::managed_memory_resource &mr,
                    vecpar::config config, T1 &in_out_1, T2 &in_2, T3 &in_3,
                    T4 &in_4, T5 &in_5, Arguments &...args) {

  auto in_out_1_view = vecmem::get_data(in_out_1);
  auto in_2_view = vecmem::get_data(in_2);
  auto in_3_view = vecmem::get_data(in_3);
  auto in_4_view = vecmem::get_data(in_4);
  auto in_5_view = vecmem::get_data(in_5);

  internal::parallel_mmap_five<Algorithm, T1, T2, T3, T4, T5, Arguments...>(
      config, in_out_1.size(), algorithm, in_out_1_view, in_2_view, in_3_view,
      in_4_view, in_5_view, args...);
  return in_out_1;
}

/// mmap generic delegator
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
