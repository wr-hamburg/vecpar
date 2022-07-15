#ifndef VECPAR_CUDA_HOSTMEM_HPP
#define VECPAR_CUDA_HOSTMEM_HPP

#include "vecpar/cuda/detail/common/cuda_utils.hpp"
#include <vecmem/containers/data/vector_view.hpp>
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/cuda/managed_memory_resource.hpp>
#include <vecmem/memory/host_memory_resource.hpp>

#include "internal.hpp"

namespace vecpar::cuda {
/*
    template <typename... T>
    std::tuple<std::conditional_t<(std::is_object<T>::value && Iterable<T>),
            vecmem::data::vector_view<value_type_t<T>>, T>...>
    get_view_of_copied_container_or_obj(T&... obj) {
        return {([](T& i) {
            if constexpr (Iterable<T>) {
                auto buffer = internal::copy.to(vecmem::get_data(i),
internal::d_mem, vecmem::copy::type::host_to_device); auto view =
vecmem::get_data(buffer); return view; } else { return i;
            }
        }(obj))...};

    }

template <typename Algorithm,
          typename R = typename Algorithm::intermediate_result_t, typename T,
          typename... Arguments>
requires vecpar::detail::is_map<Algorithm, R, T, Arguments...> R &
parallel_map(Algorithm algorithm, vecmem::host_memory_resource &mr,
             vecpar::config config, T &data, Arguments&... args) {

  // copy input data from host to device
    auto data_buffer = internal::copy.to(vecmem::get_data(data),
internal::d_mem, vecmem::copy::type::host_to_device); auto data_view =
vecmem::get_data(data_buffer);

  // allocate map result on host and copy to device
  R *map_result = new R(data.size(), &mr);
  auto result_buffer =
      internal::copy.to(vecmem::get_data(*map_result), internal::d_mem,
                        vecmem::copy::type::host_to_device);
  auto result_view = vecmem::get_data(result_buffer);

    auto input = get_view_of_copied_container_or_obj(args...);

    auto fn = [&]<typename... P>(P&... params) {
        return internal::parallel_map<Algorithm, R, T, Arguments...>(
                config, data.size(), algorithm, result_view, data_view,
params...);
    };

    std::apply(fn, input);

    internal::copy(result_buffer, *map_result,
                 vecmem::copy::type::device_to_host);

    return *map_result;
}

template <typename Algorithm,
          typename R = typename Algorithm::intermediate_result_t, typename T,
          typename... Arguments>
requires vecpar::detail::is_map<Algorithm, R, T, Arguments...> R &
parallel_map(Algorithm algorithm, vecmem::host_memory_resource &mr, T &data,
             Arguments&... args) {
  return parallel_map(algorithm, mr, cuda::getDefaultConfig(data.size()), data,
                      args...);
}

template <typename Algorithm,
          typename R = typename Algorithm::intermediate_result_t, typename T,
          typename... Arguments>
requires vecpar::detail::is_mmap<Algorithm, R, Arguments...> R &
parallel_map(Algorithm algorithm,
             __attribute__((unused)) vecmem::host_memory_resource &mr,
             vecpar::config config, T &data, Arguments&... args) {

    // copy input data from host to device
  auto data_buffer = internal::copy.to(vecmem::get_data(data), internal::d_mem,
                                       vecmem::copy::type::host_to_device);
  auto data_view = vecmem::get_data(data_buffer);

  auto input = get_view_of_copied_container_or_obj(args...);

  auto fn = [&]<typename... P>(P&... params) {
        return internal::parallel_mmap<Algorithm, R, Arguments...>(
                config, data.size(), algorithm, data_view, params...);
  };

  std::apply(fn, input);

  internal::copy(data_buffer, data, vecmem::copy::type::device_to_host);
  return data;
}

template <typename Algorithm,
          typename R = typename Algorithm::intermediate_result_t, typename T,
          typename... Arguments>
requires vecpar::detail::is_mmap<Algorithm, R, Arguments...> R &
parallel_map(Algorithm algorithm,
             __attribute__((unused)) vecmem::host_memory_resource &mr, T &data,
             Arguments&... args) {
  return parallel_map(algorithm, mr, cuda::getDefaultConfig(data.size()), data,
                      args...);
}
*/
/// horrible wrappers

template <typename Algorithm,
          typename R = typename Algorithm::intermediate_result_t, typename T,
          typename... Arguments>
requires vecpar::detail::is_map_1<Algorithm, R, T, Arguments...> R &
parallel_map(Algorithm algorithm, vecmem::host_memory_resource &mr,
             vecpar::config config, T &data, Arguments &...args) {

  // copy input data from host to device
  auto data_buffer = internal::copy.to(vecmem::get_data(data), internal::d_mem,
                                       vecmem::copy::type::host_to_device);
  auto data_view = vecmem::get_data(data_buffer);

  // allocate map result on host and copy to device
  R *map_result = new R(data.size(), &mr);
  auto result_buffer =
      internal::copy.to(vecmem::get_data(*map_result), internal::d_mem,
                        vecmem::copy::type::host_to_device);
  auto result_view = vecmem::get_data(result_buffer);

  internal::parallel_map<Algorithm, R, T, Arguments...>(
      config, data.size(), algorithm, result_view, data_view, args...);

  internal::copy(result_buffer, *map_result,
                 vecmem::copy::type::device_to_host);

  return *map_result;
}

template <typename Algorithm,
          typename R = typename Algorithm::intermediate_result_t, typename T1,
          typename T2, typename... Arguments>
requires vecpar::detail::is_map_2<Algorithm, R, T1, T2, Arguments...> R &
parallel_map(Algorithm algorithm, vecmem::host_memory_resource &mr,
             vecpar::config config, T1 &in_1, T2 &in_2, Arguments &...args) {

  // copy input data from host to device
  auto buff_1 = internal::copy.to(vecmem::get_data(in_1), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_1 = vecmem::get_data(buff_1);

  auto buf_2 = internal::copy.to(vecmem::get_data(in_2), internal::d_mem,
                                 vecmem::copy::type::host_to_device);
  auto data_view_2 = vecmem::get_data(buf_2);

  // allocate map result on host and copy to device
  R *map_result = new R(in_1.size(), &mr);
  auto result_buffer =
      internal::copy.to(vecmem::get_data(*map_result), internal::d_mem,
                        vecmem::copy::type::host_to_device);
  auto result_view = vecmem::get_data(result_buffer);

  internal::parallel_map<Algorithm, R, T1, T2, Arguments...>(
      config, in_1.size(), algorithm, result_view, data_view_1, data_view_2,
      args...);

  internal::copy(result_buffer, *map_result,
                 vecmem::copy::type::device_to_host);

  return *map_result;
}

template <typename Algorithm, typename R = typename Algorithm::result_t,
          typename T1, typename T2, typename T3, typename... Arguments>
requires vecpar::detail::is_map_3<Algorithm, R, T1, T2, T3, Arguments...>
void parallel_map(Algorithm algorithm, vecmem::host_memory_resource &mr,
                  vecpar::config config, T1 &in_1, T2 &in_2, T3 &in_3,
                  Arguments... args) {

  auto buff_1 = internal::copy.to(vecmem::get_data(in_1), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_1 = vecmem::get_data(buff_1);

  auto buf_2 = internal::copy.to(vecmem::get_data(in_2), internal::d_mem,
                                 vecmem::copy::type::host_to_device);
  auto data_view_2 = vecmem::get_data(buf_2);

  auto buf_3 = internal::copy.to(vecmem::get_data(in_3), internal::d_mem,
                                 vecmem::copy::type::host_to_device);
  auto data_view_3 = vecmem::get_data(buf_3);

  // allocate map result on host and copy to device
  R *map_result = new R(in_1.size(), &mr);
  auto result_buffer =
      internal::copy.to(vecmem::get_data(*map_result), internal::d_mem,
                        vecmem::copy::type::host_to_device);
  auto result_view = vecmem::get_data(result_buffer);

  internal::parallel_map<Algorithm, R, T1, T2, T3, Arguments...>(
      config, in_1.size(), algorithm, result_view, data_view_1, data_view_2,
      data_view_3, args...);
  internal::copy(result_buffer, *map_result,
                 vecmem::copy::type::device_to_host);

  return *map_result;
}

template <typename Algorithm, typename R = typename Algorithm::result_t,
          typename T1, typename T2, typename T3, typename T4,
          typename... Arguments>
requires vecpar::detail::is_map_4<Algorithm, R, T1, T2, T3, T4, Arguments...>
void parallel_map(Algorithm algorithm, vecmem::host_memory_resource &mr,
                  vecpar::config config, T1 &in_1, T2 &in_2, T3 &in_3, T4 &in_4,
                  Arguments... args) {

  auto buff_1 = internal::copy.to(vecmem::get_data(in_1), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_1 = vecmem::get_data(buff_1);

  auto buf_2 = internal::copy.to(vecmem::get_data(in_2), internal::d_mem,
                                 vecmem::copy::type::host_to_device);
  auto data_view_2 = vecmem::get_data(buf_2);

  auto buf_3 = internal::copy.to(vecmem::get_data(in_3), internal::d_mem,
                                 vecmem::copy::type::host_to_device);
  auto data_view_3 = vecmem::get_data(buf_3);

  auto buf_4 = internal::copy.to(vecmem::get_data(in_4), internal::d_mem,
                                 vecmem::copy::type::host_to_device);
  auto data_view_4 = vecmem::get_data(buf_4);

  // allocate map result on host and copy to device
  R *map_result = new R(in_1.size(), &mr);
  auto result_buffer =
      internal::copy.to(vecmem::get_data(*map_result), internal::d_mem,
                        vecmem::copy::type::host_to_device);
  auto result_view = vecmem::get_data(result_buffer);

  internal::parallel_map<Algorithm, R, T1, T2, T3, T4, Arguments...>(
      config, in_1.size(), algorithm, result_view, data_view_1, data_view_2,
      data_view_3, data_view_4, args...);
  internal::copy(result_buffer, *map_result,
                 vecmem::copy::type::device_to_host);

  return *map_result;
}

template <typename Algorithm, typename R = typename Algorithm::result_t,
          typename T1, typename T2, typename T3, typename T4, typename T5,
          typename... Arguments>
requires vecpar::detail::is_map_5<Algorithm, R, T1, T2, T3, T4, T5,
                                  Arguments...>
void parallel_map(Algorithm algorithm, vecmem::host_memory_resource &mr,
                  vecpar::config config, T1 &in_1, T2 &in_2, T3 &in_3, T4 &in_4,
                  T5 &in_5, Arguments... args) {

  auto buff_1 = internal::copy.to(vecmem::get_data(in_1), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_1 = vecmem::get_data(buff_1);

  auto buf_2 = internal::copy.to(vecmem::get_data(in_2), internal::d_mem,
                                 vecmem::copy::type::host_to_device);
  auto data_view_2 = vecmem::get_data(buf_2);

  auto buf_3 = internal::copy.to(vecmem::get_data(in_3), internal::d_mem,
                                 vecmem::copy::type::host_to_device);
  auto data_view_3 = vecmem::get_data(buf_3);

  auto buf_4 = internal::copy.to(vecmem::get_data(in_4), internal::d_mem,
                                 vecmem::copy::type::host_to_device);
  auto data_view_4 = vecmem::get_data(buf_4);

  auto buf_5 = internal::copy.to(vecmem::get_data(in_5), internal::d_mem,
                                 vecmem::copy::type::host_to_device);
  auto data_view_5 = vecmem::get_data(buf_5);

  // allocate map result on host and copy to device
  R *map_result = new R(in_1.size(), &mr);
  auto result_buffer =
      internal::copy.to(vecmem::get_data(*map_result), internal::d_mem,
                        vecmem::copy::type::host_to_device);
  auto result_view = vecmem::get_data(result_buffer);

  internal::parallel_map<Algorithm, R, T1, T2, T3, T4, T5, Arguments...>(
      config, in_1.size(), algorithm, result_view, data_view_1, data_view_2,
      data_view_3, data_view_4, data_view_5, args...);
  internal::copy(result_buffer, *map_result,
                 vecmem::copy::type::device_to_host);

  return *map_result;
}

template <typename Algorithm,
          typename R = typename Algorithm::intermediate_result_t, typename T,
          typename... Arguments>
requires vecpar::detail::is_map<Algorithm, R, T, Arguments...> R &
parallel_map(Algorithm algorithm, vecmem::host_memory_resource &mr, T &data,
             Arguments &...args) {
  return parallel_map(algorithm, mr, cuda::getDefaultConfig(data.size()), data,
                      args...);
}

template <typename Algorithm,
          typename R = typename Algorithm::intermediate_result_t, typename T,
          typename... Arguments>
requires vecpar::detail::is_mmap_1<Algorithm, R, Arguments...> R &
parallel_map(Algorithm algorithm,
             __attribute__((unused)) vecmem::host_memory_resource &mr,
             vecpar::config config, T &data, Arguments &...args) {

  // copy input data from host to device
  auto data_buffer = internal::copy.to(vecmem::get_data(data), internal::d_mem,
                                       vecmem::copy::type::host_to_device);
  auto data_view = vecmem::get_data(data_buffer);

  internal::parallel_mmap<Algorithm, R, Arguments...>(
      config, data.size(), algorithm, data_view, args...);

  internal::copy(data_buffer, data, vecmem::copy::type::device_to_host);
  return data;
}

template <typename Algorithm,
          typename R = typename Algorithm::intermediate_result_t, typename T1,
          typename T2, typename... Arguments>
requires vecpar::detail::is_mmap_2<Algorithm, T1, T2, Arguments...> R &
parallel_map(Algorithm algorithm,
             __attribute__((unused)) vecmem::host_memory_resource &mr,
             vecpar::config config, T1 &in_1, T2 &in_2, Arguments &...args) {

  // copy input data from host to device
  auto buff_1 = internal::copy.to(vecmem::get_data(in_1), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_1 = vecmem::get_data(buff_1);

  auto buf_2 = internal::copy.to(vecmem::get_data(in_2), internal::d_mem,
                                 vecmem::copy::type::host_to_device);
  auto data_view_2 = vecmem::get_data(buf_2);

  internal::parallel_mmap<Algorithm, T1, T2, Arguments...>(
      config, in_1.size(), algorithm, data_view_1, data_view_2, args...);

  internal::copy(buff_1, in_1, vecmem::copy::type::device_to_host);

  return in_1;
}

template <typename Algorithm, typename R = typename Algorithm::result_t,
          typename T1, typename T2, typename T3, typename... Arguments>
requires vecpar::detail::is_mmap_3<Algorithm, T1, T2, T3, Arguments...>
void parallel_map(Algorithm algorithm,
                  __attribute__((unused)) vecmem::host_memory_resource &mr,
                  vecpar::config config, T1 &in_1, T2 &in_2, T3 &in_3,
                  Arguments... args) {

  auto buff_1 = internal::copy.to(vecmem::get_data(in_1), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_1 = vecmem::get_data(buff_1);

  auto buf_2 = internal::copy.to(vecmem::get_data(in_2), internal::d_mem,
                                 vecmem::copy::type::host_to_device);
  auto data_view_2 = vecmem::get_data(buf_2);

  auto buf_3 = internal::copy.to(vecmem::get_data(in_3), internal::d_mem,
                                 vecmem::copy::type::host_to_device);
  auto data_view_3 = vecmem::get_data(buf_3);

  internal::parallel_mmap<Algorithm, T1, T2, T3, Arguments...>(
      config, in_1.size(), algorithm, data_view_1, data_view_2, data_view_3,
      args...);
  internal::copy(buff_1, in_1, vecmem::copy::type::device_to_host);
  return in_1;
}

template <typename Algorithm, typename R = typename Algorithm::result_t,
          typename T1, typename T2, typename T3, typename T4,
          typename... Arguments>
requires vecpar::detail::is_mmap_4<Algorithm, T1, T2, T3, T4, Arguments...>
void parallel_map(Algorithm algorithm,
                  __attribute__((unused)) vecmem::host_memory_resource &mr,
                  vecpar::config config, T1 &in_1, T2 &in_2, T3 &in_3, T4 &in_4,
                  Arguments... args) {

  auto buff_1 = internal::copy.to(vecmem::get_data(in_1), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_1 = vecmem::get_data(buff_1);

  auto buf_2 = internal::copy.to(vecmem::get_data(in_2), internal::d_mem,
                                 vecmem::copy::type::host_to_device);
  auto data_view_2 = vecmem::get_data(buf_2);

  auto buf_3 = internal::copy.to(vecmem::get_data(in_3), internal::d_mem,
                                 vecmem::copy::type::host_to_device);
  auto data_view_3 = vecmem::get_data(buf_3);

  auto buf_4 = internal::copy.to(vecmem::get_data(in_4), internal::d_mem,
                                 vecmem::copy::type::host_to_device);
  auto data_view_4 = vecmem::get_data(buf_4);

  internal::parallel_mmap<Algorithm, T1, T2, T3, T4, Arguments...>(
      config, in_1.size(), algorithm, data_view_1, data_view_2, data_view_3,
      data_view_4, args...);
  internal::copy(buff_1, in_1, vecmem::copy::type::device_to_host);
  return in_1;
}

template <typename Algorithm, typename R = typename Algorithm::result_t,
          typename T1, typename T2, typename T3, typename T4, typename T5,
          typename... Arguments>
requires vecpar::detail::is_mmap_5<Algorithm, T1, T2, T3, T4, T5, Arguments...>
void parallel_map(Algorithm algorithm,
                  __attribute__((unused)) vecmem::host_memory_resource &mr,
                  vecpar::config config, T1 &in_1, T2 &in_2, T3 &in_3, T4 &in_4,
                  T5 &in_5, Arguments... args) {

  auto buff_1 = internal::copy.to(vecmem::get_data(in_1), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_1 = vecmem::get_data(buff_1);

  auto buf_2 = internal::copy.to(vecmem::get_data(in_2), internal::d_mem,
                                 vecmem::copy::type::host_to_device);
  auto data_view_2 = vecmem::get_data(buf_2);

  auto buf_3 = internal::copy.to(vecmem::get_data(in_3), internal::d_mem,
                                 vecmem::copy::type::host_to_device);
  auto data_view_3 = vecmem::get_data(buf_3);

  auto buf_4 = internal::copy.to(vecmem::get_data(in_4), internal::d_mem,
                                 vecmem::copy::type::host_to_device);
  auto data_view_4 = vecmem::get_data(buf_4);

  auto buf_5 = internal::copy.to(vecmem::get_data(in_5), internal::d_mem,
                                 vecmem::copy::type::host_to_device);
  auto data_view_5 = vecmem::get_data(buf_5);

  internal::parallel_map<Algorithm, T1, T2, T3, T4, T5, Arguments...>(
      config, in_1.size(), algorithm, data_view_1, data_view_2, data_view_3,
      data_view_4, data_view_5, args...);
  internal::copy(buff_1, in_1, vecmem::copy::type::device_to_host);
  return in_1;
}

template <typename Algorithm,
          typename R = typename Algorithm::intermediate_result_t, typename T,
          typename... Arguments>
requires vecpar::detail::is_mmap<Algorithm, T, Arguments...> R &
parallel_map(Algorithm algorithm, vecmem::host_memory_resource &mr, T &data,
             Arguments &...args) {
  return parallel_map(algorithm, mr, cuda::getDefaultConfig(data.size()), data,
                      args...);
}

template <typename Algorithm, typename R>
typename R::value_type &parallel_reduce(Algorithm algorithm,
                                        __attribute__((unused))
                                        vecmem::host_memory_resource &mr,
                                        R &data) {

  // copy input data from host to device
  auto data_buffer = internal::copy.to(vecmem::get_data(data), internal::d_mem,
                                       vecmem::copy::type::host_to_device);
  auto data_view = vecmem::get_data(data_buffer);

  // TODO: return a pointer to host from here
  typename R::value_type *d_result;
  cudaMallocManaged(&d_result, sizeof(typename R::value_type));
  memset(d_result, 0, sizeof(typename R::value_type));

  internal::parallel_reduce(data.size(), algorithm, d_result, data_view);
  return *d_result;
}

template <typename Algorithm, typename R>
R &parallel_filter(Algorithm algorithm, vecmem::host_memory_resource &mr,
                   R &data) {

  // copy input data from host to device
  auto data_buffer = internal::copy.to(vecmem::get_data(data), internal::d_mem,
                                       vecmem::copy::type::host_to_device);
  auto data_view = vecmem::get_data(data_buffer);

  // allocate result on host and device
  R *result = new R(data.size(), &mr);

  auto result_buffer =
      internal::copy.to(vecmem::get_data(*result), internal::d_mem,
                        vecmem::copy::type::host_to_device);
  auto result_view = vecmem::get_data(result_buffer);

  int *idx; // global index
  CHECK_ERROR(cudaMallocManaged((void **)&idx, sizeof(int)))
  *idx = 0;

  internal::parallel_filter(data.size(), algorithm, idx, result_view,
                            data_view);

  // copy back results
  internal::copy(result_buffer, *result, vecmem::copy::type::device_to_host);
  result->resize(*idx);

  // release the memory allocated for the index
  CHECK_ERROR(cudaFree(idx))

  return *result;
}
/*
template <class Algorithm, typename Result = typename Algorithm::result_t,
         typename R = typename Algorithm::intermediate_result_t, typename T,
         typename... Arguments>
requires vecpar::algorithm::is_map_reduce<Algorithm, Result, R, T, Arguments...>
   Result &
   parallel_map_reduce(Algorithm algorithm, vecmem::host_memory_resource &mr,
                       vecpar::config config, T &data, Arguments&... args) {

//  copy input data from host to device
   auto data_buffer = internal::copy.to(vecmem::get_data(data), internal::d_mem,
                                        vecmem::copy::type::host_to_device);
   auto data_view = vecmem::get_data(data_buffer);

 // allocate temp map result on host and copy to device
 R *map_result = new R(data.size(), &mr);
 auto result_buffer =
     internal::copy.to(vecmem::get_data(*map_result), internal::d_mem,
                       vecmem::copy::type::host_to_device);
 auto result_view = vecmem::get_data(result_buffer);
 auto input = get_view_of_copied_container_or_obj(args...);

   auto fn = [&]<typename... P>(P&... params) {
       return internal::parallel_map<Algorithm, R, T, Arguments...>(
               config, data.size(), algorithm, result_view, data_view,
params...);
   };

 std::apply(fn, input);

 Result *result = (Result *)malloc(sizeof(Result));
 Result *d_result;

 CHECK_ERROR(cudaMalloc((void **)&d_result, sizeof(Result)))
 CHECK_ERROR(cudaMemset(d_result, 0, sizeof(Result)))

 internal::parallel_reduce(data.size(), algorithm, d_result, result_view);

 CHECK_ERROR(
     cudaMemcpy(result, d_result, sizeof(Result), cudaMemcpyDeviceToHost))
 CHECK_ERROR(cudaFree(d_result))

 return *result;
}
*/
template <class Algorithm, typename Result = typename Algorithm::result_t,
          typename R = typename Algorithm::intermediate_result_t, typename T,
          typename... Arguments>
requires vecpar::algorithm::is_map_reduce_1<Algorithm, Result, R, T,
                                            Arguments...>
    Result &
    parallel_map_reduce(Algorithm algorithm, vecmem::host_memory_resource &mr,
                        vecpar::config config, T &in_1, Arguments &...args) {

  // copy input data from host to device
  auto buff_1 = internal::copy.to(vecmem::get_data(in_1), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_1 = vecmem::get_data(buff_1);

  // allocate map result on host and copy to device
  R *map_result = new R(in_1.size(), &mr);
  auto result_buffer =
      internal::copy.to(vecmem::get_data(*map_result), internal::d_mem,
                        vecmem::copy::type::host_to_device);
  auto result_view = vecmem::get_data(result_buffer);

  internal::parallel_map<Algorithm, R, T, Arguments...>(
      config, in_1.size(), algorithm, result_view, data_view_1, args...);

  Result *result = (Result *)malloc(sizeof(Result));
  Result *d_result;

  CHECK_ERROR(cudaMalloc((void **)&d_result, sizeof(Result)))
  CHECK_ERROR(cudaMemset(d_result, 0, sizeof(Result)))

  internal::parallel_reduce(in_1.size(), algorithm, d_result, result_view);

  CHECK_ERROR(
      cudaMemcpy(result, d_result, sizeof(Result), cudaMemcpyDeviceToHost))
  CHECK_ERROR(cudaFree(d_result))

  return *result;
}

template <class Algorithm, typename Result = typename Algorithm::result_t,
          typename R = typename Algorithm::intermediate_result_t, typename T1,
          typename T2, typename... Arguments>
requires vecpar::algorithm::is_map_reduce_2<Algorithm, Result, R, T1, T2,
                                            Arguments...>
    Result &parallel_map_reduce(Algorithm algorithm,
                                vecmem::host_memory_resource &mr,
                                vecpar::config config, T1 &in_1, T2 &in_2,
                                Arguments &...args) {

  // copy input data from host to device
  auto buff_1 = internal::copy.to(vecmem::get_data(in_1), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_1 = vecmem::get_data(buff_1);

  auto buff_2 = internal::copy.to(vecmem::get_data(in_2), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_2 = vecmem::get_data(buff_2);

  // allocate map result on host and copy to device
  R *map_result = new R(in_1.size(), &mr);
  auto result_buffer =
      internal::copy.to(vecmem::get_data(*map_result), internal::d_mem,
                        vecmem::copy::type::host_to_device);
  auto result_view = vecmem::get_data(result_buffer);

  internal::parallel_map<Algorithm, R, T1, T2, Arguments...>(
      config, in_1.size(), algorithm, result_view, data_view_1, data_view_2,
      args...);

  Result *result = (Result *)malloc(sizeof(Result));
  Result *d_result;

  CHECK_ERROR(cudaMalloc((void **)&d_result, sizeof(Result)))
  CHECK_ERROR(cudaMemset(d_result, 0, sizeof(Result)))

  internal::parallel_reduce(in_1.size(), algorithm, d_result, result_view);

  CHECK_ERROR(
      cudaMemcpy(result, d_result, sizeof(Result), cudaMemcpyDeviceToHost))
  CHECK_ERROR(cudaFree(d_result))

  return *result;
}

template <class Algorithm, typename Result = typename Algorithm::result_t,
          typename R = typename Algorithm::intermediate_result_t, typename T1,
          typename T2, typename T3, typename... Arguments>
requires vecpar::algorithm::is_map_reduce_3<Algorithm, Result, R, T1, T2, T3,
                                            Arguments...>
    Result &parallel_map_reduce(Algorithm algorithm,
                                vecmem::host_memory_resource &mr,
                                vecpar::config config, T1 &in_1, T2 &in_2,
                                T3 &in_3, Arguments &...args) {

  // copy input data from host to device
  auto buff_1 = internal::copy.to(vecmem::get_data(in_1), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_1 = vecmem::get_data(buff_1);

  auto buff_2 = internal::copy.to(vecmem::get_data(in_2), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_2 = vecmem::get_data(buff_2);

  auto buff_3 = internal::copy.to(vecmem::get_data(in_3), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_3 = vecmem::get_data(buff_3);

  // allocate map result on host and copy to device
  R *map_result = new R(in_1.size(), &mr);
  auto result_buffer =
      internal::copy.to(vecmem::get_data(*map_result), internal::d_mem,
                        vecmem::copy::type::host_to_device);
  auto result_view = vecmem::get_data(result_buffer);

  internal::parallel_map<Algorithm, R, T1, T2, T3, Arguments...>(
      config, in_1.size(), algorithm, result_view, data_view_1, data_view_2,
      data_view_3, args...);

  Result *result = (Result *)malloc(sizeof(Result));
  Result *d_result;

  CHECK_ERROR(cudaMalloc((void **)&d_result, sizeof(Result)))
  CHECK_ERROR(cudaMemset(d_result, 0, sizeof(Result)))

  internal::parallel_reduce(in_1.size(), algorithm, d_result, result_view);

  CHECK_ERROR(
      cudaMemcpy(result, d_result, sizeof(Result), cudaMemcpyDeviceToHost))
  CHECK_ERROR(cudaFree(d_result))

  return *result;
}

template <class Algorithm, typename Result = typename Algorithm::result_t,
          typename R = typename Algorithm::intermediate_result_t, typename T1,
          typename T2, typename T3, typename T4, typename... Arguments>
requires vecpar::algorithm::is_map_reduce_4<Algorithm, Result, R, T1, T2, T3,
                                            T4, Arguments...>
    Result &parallel_map_reduce(Algorithm algorithm,
                                vecmem::host_memory_resource &mr,
                                vecpar::config config, T1 &in_1, T2 &in_2,
                                T3 &in_3, T4 &in_4, Arguments &...args) {

  // copy input data from host to device
  auto buff_1 = internal::copy.to(vecmem::get_data(in_1), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_1 = vecmem::get_data(buff_1);

  auto buff_2 = internal::copy.to(vecmem::get_data(in_2), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_2 = vecmem::get_data(buff_2);

  auto buff_3 = internal::copy.to(vecmem::get_data(in_3), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_3 = vecmem::get_data(buff_3);

  auto buff_4 = internal::copy.to(vecmem::get_data(in_4), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_4 = vecmem::get_data(buff_4);

  // allocate map result on host and copy to device
  R *map_result = new R(in_1.size(), &mr);
  auto result_buffer =
      internal::copy.to(vecmem::get_data(*map_result), internal::d_mem,
                        vecmem::copy::type::host_to_device);
  auto result_view = vecmem::get_data(result_buffer);

  internal::parallel_map<Algorithm, R, T1, T2, T3, T4, Arguments...>(
      config, in_1.size(), algorithm, result_view, data_view_1, data_view_2,
      data_view_3, data_view_4, args...);

  Result *result = (Result *)malloc(sizeof(Result));
  Result *d_result;

  CHECK_ERROR(cudaMalloc((void **)&d_result, sizeof(Result)))
  CHECK_ERROR(cudaMemset(d_result, 0, sizeof(Result)))

  internal::parallel_reduce(in_1.size(), algorithm, d_result, result_view);

  CHECK_ERROR(
      cudaMemcpy(result, d_result, sizeof(Result), cudaMemcpyDeviceToHost))
  CHECK_ERROR(cudaFree(d_result))

  return *result;
}

template <class Algorithm, typename Result = typename Algorithm::result_t,
          typename R = typename Algorithm::intermediate_result_t, typename T1,
          typename T2, typename T3, typename T4, typename T5,
          typename... Arguments>
requires vecpar::algorithm::is_map_reduce_5<Algorithm, Result, R, T1, T2, T3,
                                            T4, T5, Arguments...>
    Result &
    parallel_map_reduce(Algorithm algorithm, vecmem::host_memory_resource &mr,
                        vecpar::config config, T1 &in_1, T2 &in_2, T3 &in_3,
                        T4 &in_4, T5 &in_5, Arguments &...args) {

  // copy input data from host to device
  auto buff_1 = internal::copy.to(vecmem::get_data(in_1), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_1 = vecmem::get_data(buff_1);

  auto buff_2 = internal::copy.to(vecmem::get_data(in_2), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_2 = vecmem::get_data(buff_2);

  auto buff_3 = internal::copy.to(vecmem::get_data(in_3), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_3 = vecmem::get_data(buff_3);

  auto buff_4 = internal::copy.to(vecmem::get_data(in_4), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_4 = vecmem::get_data(buff_4);

  auto buff_5 = internal::copy.to(vecmem::get_data(in_5), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_5 = vecmem::get_data(buff_5);

  // allocate map result on host and copy to device
  R *map_result = new R(in_1.size(), &mr);
  auto result_buffer =
      internal::copy.to(vecmem::get_data(*map_result), internal::d_mem,
                        vecmem::copy::type::host_to_device);
  auto result_view = vecmem::get_data(result_buffer);

  internal::parallel_map<Algorithm, R, T1, T2, T3, T4, T5, Arguments...>(
      config, in_1.size(), algorithm, result_view, data_view_1, data_view_2,
      data_view_3, data_view_4, data_view_5, args...);

  Result *result = (Result *)malloc(sizeof(Result));
  Result *d_result;

  CHECK_ERROR(cudaMalloc((void **)&d_result, sizeof(Result)))
  CHECK_ERROR(cudaMemset(d_result, 0, sizeof(Result)))

  internal::parallel_reduce(in_1.size(), algorithm, d_result, result_view);

  CHECK_ERROR(
      cudaMemcpy(result, d_result, sizeof(Result), cudaMemcpyDeviceToHost))
  CHECK_ERROR(cudaFree(d_result))

  return *result;
}

template <class Algorithm, typename Result = typename Algorithm::result_t,
          typename R = typename Algorithm::intermediate_result_t, typename T,
          typename... Arguments>
requires vecpar::algorithm::is_map_reduce<Algorithm, Result, R, T, Arguments...>
    Result &parallel_map_reduce(Algorithm algorithm,
                                vecmem::host_memory_resource &mr, T &data,
                                Arguments &...args) {
  return parallel_map_reduce(algorithm, mr, cuda::getDefaultConfig(data.size()),
                             data, args...);
}

/*
template <class Algorithm, typename Result = typename Algorithm::result_t,
          typename R = typename Algorithm::intermediate_result_t, typename T,
          typename... Arguments>
requires vecpar::algorithm::is_mmap_reduce<Algorithm, Result, R, Arguments...>
    Result &parallel_map_reduce(Algorithm algorithm,
                                __attribute__((unused))
                                vecmem::host_memory_resource &mr,
                                vecpar::config config, T &data,
                                Arguments&... args) {
  // copy input data from host to device
  auto data_buffer = internal::copy.to(vecmem::get_data(data), internal::d_mem,
                                       vecmem::copy::type::host_to_device);
  auto data_view = vecmem::get_data(data_buffer);

  auto input = get_view_of_copied_container_or_obj(args...);

    auto fn = [&]<typename... P>(P&... params) {
        return internal::parallel_mmap<Algorithm, R, Arguments...>(
                config, data.size(), algorithm, data_view, params...);
    };

  std::apply(fn, input);

  Result *result = (Result *)malloc(sizeof(Result));
  Result *d_result;

  CHECK_ERROR(cudaMalloc((void **)&d_result, sizeof(Result)))
  CHECK_ERROR(cudaMemset(d_result, 0, sizeof(Result)))

  internal::parallel_reduce(data.size(), algorithm, d_result, data_view);

  CHECK_ERROR(
      cudaMemcpy(result, d_result, sizeof(Result), cudaMemcpyDeviceToHost))
  CHECK_ERROR(cudaFree(d_result))

  return *result;
}
*/

template <class Algorithm, typename Result = typename Algorithm::result_t,
          typename R = typename Algorithm::intermediate_result_t, typename T,
          typename... Arguments>
requires vecpar::algorithm::is_mmap_reduce_1<Algorithm, Result, T, Arguments...>
    Result &parallel_map_reduce(Algorithm algorithm,
                                __attribute__((unused))
                                vecmem::host_memory_resource &mr,
                                vecpar::config config, T &in_1,
                                Arguments &...args) {

  // copy input data from host to device
  auto buff_1 = internal::copy.to(vecmem::get_data(in_1), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_1 = vecmem::get_data(buff_1);

  internal::parallel_mmap<Algorithm, T, Arguments...>(
      config, in_1.size(), algorithm, data_view_1, args...);

  Result *result = (Result *)malloc(sizeof(Result));
  Result *d_result;

  CHECK_ERROR(cudaMalloc((void **)&d_result, sizeof(Result)))
  CHECK_ERROR(cudaMemset(d_result, 0, sizeof(Result)))

  internal::parallel_reduce(in_1.size(), algorithm, d_result, data_view_1);

  CHECK_ERROR(
      cudaMemcpy(result, d_result, sizeof(Result), cudaMemcpyDeviceToHost))
  CHECK_ERROR(cudaFree(d_result))

  return *result;
}

template <class Algorithm, typename Result = typename Algorithm::result_t,
          typename R = typename Algorithm::intermediate_result_t, typename T1,
          typename T2, typename... Arguments>
requires vecpar::algorithm::is_mmap_reduce_2<Algorithm, Result, T1, T2,
                                             Arguments...>
    Result &parallel_map_reduce(Algorithm algorithm,
                                __attribute__((unused))
                                vecmem::host_memory_resource &mr,
                                vecpar::config config, T1 &in_1, T2 &in_2,
                                Arguments &...args) {

  // copy input data from host to device
  auto buff_1 = internal::copy.to(vecmem::get_data(in_1), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_1 = vecmem::get_data(buff_1);

  auto buff_2 = internal::copy.to(vecmem::get_data(in_2), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_2 = vecmem::get_data(buff_2);

  internal::parallel_mmap<Algorithm, T1, T2, Arguments...>(
      config, in_1.size(), algorithm, data_view_1, data_view_2, args...);

  Result *result = (Result *)malloc(sizeof(Result));
  Result *d_result;

  CHECK_ERROR(cudaMalloc((void **)&d_result, sizeof(Result)))
  CHECK_ERROR(cudaMemset(d_result, 0, sizeof(Result)))

  internal::parallel_reduce(in_1.size(), algorithm, d_result, data_view_1);

  CHECK_ERROR(
      cudaMemcpy(result, d_result, sizeof(Result), cudaMemcpyDeviceToHost))
  CHECK_ERROR(cudaFree(d_result))

  return *result;
}

template <class Algorithm, typename Result = typename Algorithm::result_t,
          typename R = typename Algorithm::intermediate_result_t, typename T1,
          typename T2, typename T3, typename... Arguments>
requires vecpar::algorithm::is_mmap_reduce_3<Algorithm, Result, T1, T2, T3,
                                             Arguments...>
    Result &parallel_map_reduce(Algorithm algorithm,
                                __attribute__((unused))
                                vecmem::host_memory_resource &mr,
                                vecpar::config config, T1 &in_1, T2 &in_2,
                                T3 &in_3, Arguments &...args) {

  // copy input data from host to device
  auto buff_1 = internal::copy.to(vecmem::get_data(in_1), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_1 = vecmem::get_data(buff_1);

  auto buff_2 = internal::copy.to(vecmem::get_data(in_2), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_2 = vecmem::get_data(buff_2);

  auto buff_3 = internal::copy.to(vecmem::get_data(in_3), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_3 = vecmem::get_data(buff_3);

  internal::parallel_mmap<Algorithm, T1, T2, T3, Arguments...>(
      config, in_1.size(), algorithm, data_view_1, data_view_2, data_view_3,
      args...);

  Result *result = (Result *)malloc(sizeof(Result));
  Result *d_result;

  CHECK_ERROR(cudaMalloc((void **)&d_result, sizeof(Result)))
  CHECK_ERROR(cudaMemset(d_result, 0, sizeof(Result)))

  internal::parallel_reduce(in_1.size(), algorithm, d_result, data_view_1);

  CHECK_ERROR(
      cudaMemcpy(result, d_result, sizeof(Result), cudaMemcpyDeviceToHost))
  CHECK_ERROR(cudaFree(d_result))

  return *result;
}

template <class Algorithm, typename Result = typename Algorithm::result_t,
          typename R = typename Algorithm::intermediate_result_t, typename T1,
          typename T2, typename T3, typename T4, typename... Arguments>
requires vecpar::algorithm::is_mmap_reduce_4<Algorithm, Result, T1, T2, T3, T4,
                                             Arguments...>
    Result &parallel_map_reduce(Algorithm algorithm,
                                __attribute__((unused))
                                vecmem::host_memory_resource &mr,
                                vecpar::config config, T1 &in_1, T2 &in_2,
                                T3 &in_3, T4 &in_4, Arguments &...args) {

  // copy input data from host to device
  auto buff_1 = internal::copy.to(vecmem::get_data(in_1), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_1 = vecmem::get_data(buff_1);

  auto buff_2 = internal::copy.to(vecmem::get_data(in_2), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_2 = vecmem::get_data(buff_2);

  auto buff_3 = internal::copy.to(vecmem::get_data(in_3), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_3 = vecmem::get_data(buff_3);

  auto buff_4 = internal::copy.to(vecmem::get_data(in_4), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_4 = vecmem::get_data(buff_4);

  internal::parallel_mmap<Algorithm, T1, T2, T3, T4, Arguments...>(
      config, in_1.size(), algorithm, data_view_1, data_view_2, data_view_3,
      data_view_4, args...);

  Result *result = (Result *)malloc(sizeof(Result));
  Result *d_result;

  CHECK_ERROR(cudaMalloc((void **)&d_result, sizeof(Result)))
  CHECK_ERROR(cudaMemset(d_result, 0, sizeof(Result)))

  internal::parallel_reduce(in_1.size(), algorithm, d_result, data_view_1);

  CHECK_ERROR(
      cudaMemcpy(result, d_result, sizeof(Result), cudaMemcpyDeviceToHost))
  CHECK_ERROR(cudaFree(d_result))

  return *result;
}

template <class Algorithm, typename Result = typename Algorithm::result_t,
          typename R = typename Algorithm::intermediate_result_t, typename T1,
          typename T2, typename T3, typename T4, typename T5,
          typename... Arguments>
requires vecpar::algorithm::is_mmap_reduce_5<Algorithm, Result, T1, T2, T3, T4,
                                             T5, Arguments...>
    Result &parallel_map_reduce(Algorithm algorithm,
                                __attribute__((unused))
                                vecmem::host_memory_resource &mr,
                                vecpar::config config, T1 &in_1, T2 &in_2,
                                T3 &in_3, T4 &in_4, T5 &in_5,
                                Arguments &...args) {

  // copy input data from host to device
  auto buff_1 = internal::copy.to(vecmem::get_data(in_1), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_1 = vecmem::get_data(buff_1);

  auto buff_2 = internal::copy.to(vecmem::get_data(in_2), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_2 = vecmem::get_data(buff_2);

  auto buff_3 = internal::copy.to(vecmem::get_data(in_3), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_3 = vecmem::get_data(buff_3);

  auto buff_4 = internal::copy.to(vecmem::get_data(in_4), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_4 = vecmem::get_data(buff_4);

  auto buff_5 = internal::copy.to(vecmem::get_data(in_5), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_5 = vecmem::get_data(buff_5);

  internal::parallel_mmap<Algorithm, T1, T2, T3, T4, T5, Arguments...>(
      config, in_1.size(), algorithm, data_view_1, data_view_2, data_view_3,
      data_view_4, data_view_5, args...);

  Result *result = (Result *)malloc(sizeof(Result));
  Result *d_result;

  CHECK_ERROR(cudaMalloc((void **)&d_result, sizeof(Result)))
  CHECK_ERROR(cudaMemset(d_result, 0, sizeof(Result)))

  internal::parallel_reduce(in_1.size(), algorithm, d_result, data_view_1);

  CHECK_ERROR(
      cudaMemcpy(result, d_result, sizeof(Result), cudaMemcpyDeviceToHost))
  CHECK_ERROR(cudaFree(d_result))

  return *result;
}

template <class Algorithm, typename Result = typename Algorithm::result_t,
          typename R = typename Algorithm::intermediate_result_t, typename T,
          typename... Arguments>
requires vecpar::algorithm::is_mmap_reduce<Algorithm, Result, R, Arguments...>
    Result &parallel_map_reduce(Algorithm algorithm,
                                __attribute__((unused))
                                vecmem::host_memory_resource &mr,
                                T &data, Arguments &...args) {
  return parallel_map_reduce(algorithm, mr, cuda::getDefaultConfig(data.size()),
                             data, args...);
}

/*
template <class Algorithm, typename R, typename T, typename... Arguments>
requires vecpar::algorithm::is_map_filter<Algorithm, R, T, Arguments...> R &
parallel_map_filter(Algorithm algorithm, vecmem::host_memory_resource &mr,
                    vecpar::config config, T &data, Arguments&... args) {
  size_t size = data.size();
  // copy input data from host to device
    auto data_buffer = internal::copy.to(vecmem::get_data(data),
internal::d_mem, vecmem::copy::type::host_to_device); auto data_view =
vecmem::get_data(data_buffer);


  // allocate temp map result on host and copy to device
  R *map_result = new R(size, &mr);

  auto map_result_buffer =
      internal::copy.to(vecmem::get_data(*map_result), internal::d_mem,
                        vecmem::copy::type::host_to_device);
  auto map_result_view = vecmem::get_data(map_result_buffer);
    auto input = get_view_of_copied_container_or_obj(args...);

  auto fn = [&]<typename... P>(P&... params) {
        return internal::parallel_map<Algorithm, R, T, Arguments...>(
                config, data.size(), algorithm, map_result_view, data_view,
params...);
  };
  std::apply(fn, input);

  // allocate result on host and device
  R *result = new R(size, &mr);
  auto result_buffer =
      internal::copy.to(vecmem::get_data(*result), internal::d_mem,
                        vecmem::copy::type::host_to_device);
  auto result_view = vecmem::get_data(result_buffer);

  int *idx; // global index
  CHECK_ERROR(cudaMallocManaged((void **)&idx, sizeof(int)))
  *idx = 0;
  internal::parallel_filter(size, algorithm, idx, result_view, map_result_view);

  // copy back results
  internal::copy(result_buffer, *result, vecmem::copy::type::device_to_host);
  result->resize(*idx);

  // release the memory allocated for the index
  CHECK_ERROR(cudaFree(idx))

  return *result;
}
*/
/// TODO: filter can be improved to avoid an allocation
template <class Algorithm, typename R, typename T, typename... Arguments>
requires vecpar::algorithm::is_map_filter_1<Algorithm, R, T, Arguments...> R &
parallel_map_filter(Algorithm algorithm, vecmem::host_memory_resource &mr,
                    vecpar::config config, T &in_1, Arguments &...args) {
  size_t size = in_1.size();
  // copy input data from host to device
  auto buff_1 = internal::copy.to(vecmem::get_data(in_1), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_1 = vecmem::get_data(buff_1);

  // allocate temp map result on host and copy to device
  R *map_result = new R(size, &mr);

  auto map_result_buffer =
      internal::copy.to(vecmem::get_data(*map_result), internal::d_mem,
                        vecmem::copy::type::host_to_device);
  auto map_result_view = vecmem::get_data(map_result_buffer);

  internal::parallel_map<Algorithm, R, T, Arguments...>(
      config, in_1.size(), algorithm, map_result_view, data_view_1, args...);

  // allocate result on host and device
  R *result = new R(size, &mr);
  auto result_buffer =
      internal::copy.to(vecmem::get_data(*result), internal::d_mem,
                        vecmem::copy::type::host_to_device);
  auto result_view = vecmem::get_data(result_buffer);

  int *idx; // global index
  CHECK_ERROR(cudaMallocManaged((void **)&idx, sizeof(int)))
  *idx = 0;
  internal::parallel_filter(size, algorithm, idx, result_view, map_result_view);

  // copy back results
  internal::copy(result_buffer, *result, vecmem::copy::type::device_to_host);
  result->resize(*idx);

  // release the memory allocated for the index
  CHECK_ERROR(cudaFree(idx))

  return *result;
}
template <class Algorithm, typename R, typename T1, typename T2,
          typename... Arguments>
requires vecpar::algorithm::is_map_filter_2<Algorithm, R, T1, T2, Arguments...>
    R &parallel_map_filter(Algorithm algorithm,
                           vecmem::host_memory_resource &mr,
                           vecpar::config config, T1 &in_1, T2 &in_2,
                           Arguments &...args) {
  size_t size = in_1.size();
  // copy input data from host to device
  auto buff_1 = internal::copy.to(vecmem::get_data(in_1), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_1 = vecmem::get_data(buff_1);

  auto buff_2 = internal::copy.to(vecmem::get_data(in_2), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_2 = vecmem::get_data(buff_2);

  // allocate temp map result on host and copy to device
  R *map_result = new R(size, &mr);

  auto map_result_buffer =
      internal::copy.to(vecmem::get_data(*map_result), internal::d_mem,
                        vecmem::copy::type::host_to_device);
  auto map_result_view = vecmem::get_data(map_result_buffer);

  internal::parallel_map<Algorithm, R, T1, T2, Arguments...>(
      config, in_1.size(), algorithm, map_result_view, data_view_1, data_view_2,
      args...);

  // allocate result on host and device
  R *result = new R(size, &mr);
  auto result_buffer =
      internal::copy.to(vecmem::get_data(*result), internal::d_mem,
                        vecmem::copy::type::host_to_device);
  auto result_view = vecmem::get_data(result_buffer);

  int *idx; // global index
  CHECK_ERROR(cudaMallocManaged((void **)&idx, sizeof(int)))
  *idx = 0;
  internal::parallel_filter(size, algorithm, idx, result_view, map_result_view);

  // copy back results
  internal::copy(result_buffer, *result, vecmem::copy::type::device_to_host);
  result->resize(*idx);

  // release the memory allocated for the index
  CHECK_ERROR(cudaFree(idx))

  return *result;
}

template <class Algorithm, typename R, typename T1, typename T2, typename T3,
          typename... Arguments>
requires vecpar::algorithm::is_map_filter_3<Algorithm, R, T1, T2, T3,
                                            Arguments...>
    R &parallel_map_filter(Algorithm algorithm,
                           vecmem::host_memory_resource &mr,
                           vecpar::config config, T1 &in_1, T2 &in_2, T3 &in_3,
                           Arguments &...args) {
  size_t size = in_1.size();
  // copy input data from host to device
  auto buff_1 = internal::copy.to(vecmem::get_data(in_1), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_1 = vecmem::get_data(buff_1);

  auto buff_2 = internal::copy.to(vecmem::get_data(in_2), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_2 = vecmem::get_data(buff_2);

  auto buff_3 = internal::copy.to(vecmem::get_data(in_3), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_3 = vecmem::get_data(buff_3);

  // allocate temp map result on host and copy to device
  R *map_result = new R(size, &mr);

  auto map_result_buffer =
      internal::copy.to(vecmem::get_data(*map_result), internal::d_mem,
                        vecmem::copy::type::host_to_device);
  auto map_result_view = vecmem::get_data(map_result_buffer);

  internal::parallel_map<Algorithm, R, T1, T2, T3, Arguments...>(
      config, in_1.size(), algorithm, map_result_view, data_view_1, data_view_2,
      data_view_3, args...);

  // allocate result on host and device
  R *result = new R(size, &mr);
  auto result_buffer =
      internal::copy.to(vecmem::get_data(*result), internal::d_mem,
                        vecmem::copy::type::host_to_device);
  auto result_view = vecmem::get_data(result_buffer);

  int *idx; // global index
  CHECK_ERROR(cudaMallocManaged((void **)&idx, sizeof(int)))
  *idx = 0;
  internal::parallel_filter(size, algorithm, idx, result_view, map_result_view);

  // copy back results
  internal::copy(result_buffer, *result, vecmem::copy::type::device_to_host);
  result->resize(*idx);

  // release the memory allocated for the index
  CHECK_ERROR(cudaFree(idx))

  return *result;
}

template <class Algorithm, typename R, typename T1, typename T2, typename T3,
          typename T4, typename... Arguments>
requires vecpar::algorithm::is_map_filter_4<Algorithm, R, T1, T2, T3, T4,
                                            Arguments...>
    R &parallel_map_filter(Algorithm algorithm,
                           vecmem::host_memory_resource &mr,
                           vecpar::config config, T1 &in_1, T2 &in_2, T3 &in_3,
                           T4 &in_4, Arguments &...args) {
  size_t size = in_1.size();
  // copy input data from host to device
  auto buff_1 = internal::copy.to(vecmem::get_data(in_1), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_1 = vecmem::get_data(buff_1);

  auto buff_2 = internal::copy.to(vecmem::get_data(in_2), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_2 = vecmem::get_data(buff_2);

  auto buff_3 = internal::copy.to(vecmem::get_data(in_3), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_3 = vecmem::get_data(buff_3);

  auto buff_4 = internal::copy.to(vecmem::get_data(in_4), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_4 = vecmem::get_data(buff_4);

  // allocate temp map result on host and copy to device
  R *map_result = new R(size, &mr);

  auto map_result_buffer =
      internal::copy.to(vecmem::get_data(*map_result), internal::d_mem,
                        vecmem::copy::type::host_to_device);
  auto map_result_view = vecmem::get_data(map_result_buffer);

  internal::parallel_map<Algorithm, R, T1, T2, T3, T4, Arguments...>(
      config, in_1.size(), algorithm, map_result_view, data_view_1, data_view_2,
      data_view_3, data_view_4, args...);

  // allocate result on host and device
  R *result = new R(size, &mr);
  auto result_buffer =
      internal::copy.to(vecmem::get_data(*result), internal::d_mem,
                        vecmem::copy::type::host_to_device);
  auto result_view = vecmem::get_data(result_buffer);

  int *idx; // global index
  CHECK_ERROR(cudaMallocManaged((void **)&idx, sizeof(int)))
  *idx = 0;
  internal::parallel_filter(size, algorithm, idx, result_view, map_result_view);

  // copy back results
  internal::copy(result_buffer, *result, vecmem::copy::type::device_to_host);
  result->resize(*idx);

  // release the memory allocated for the index
  CHECK_ERROR(cudaFree(idx))

  return *result;
}

template <class Algorithm, typename R, typename T1, typename T2, typename T3,
          typename T4, typename T5, typename... Arguments>
requires vecpar::algorithm::is_map_filter_5<Algorithm, R, T1, T2, T3, T4, T5,
                                            Arguments...>
    R &parallel_map_filter(Algorithm algorithm,
                           vecmem::host_memory_resource &mr,
                           vecpar::config config, T1 &in_1, T2 &in_2, T3 &in_3,
                           T4 &in_4, T5 &in_5, Arguments &...args) {
  size_t size = in_1.size();
  // copy input data from host to device
  auto buff_1 = internal::copy.to(vecmem::get_data(in_1), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_1 = vecmem::get_data(buff_1);

  auto buff_2 = internal::copy.to(vecmem::get_data(in_2), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_2 = vecmem::get_data(buff_2);

  auto buff_3 = internal::copy.to(vecmem::get_data(in_3), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_3 = vecmem::get_data(buff_3);

  auto buff_4 = internal::copy.to(vecmem::get_data(in_4), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_4 = vecmem::get_data(buff_4);

  auto buff_5 = internal::copy.to(vecmem::get_data(in_5), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_5 = vecmem::get_data(buff_5);

  // allocate temp map result on host and copy to device
  R *map_result = new R(size, &mr);

  auto map_result_buffer =
      internal::copy.to(vecmem::get_data(*map_result), internal::d_mem,
                        vecmem::copy::type::host_to_device);
  auto map_result_view = vecmem::get_data(map_result_buffer);

  internal::parallel_map<Algorithm, R, T1, T2, T3, T4, T5, Arguments...>(
      config, in_1.size(), algorithm, map_result_view, data_view_1, data_view_2,
      data_view_3, data_view_4, data_view_5, args...);

  // allocate result on host and device
  R *result = new R(size, &mr);
  auto result_buffer =
      internal::copy.to(vecmem::get_data(*result), internal::d_mem,
                        vecmem::copy::type::host_to_device);
  auto result_view = vecmem::get_data(result_buffer);

  int *idx; // global index
  CHECK_ERROR(cudaMallocManaged((void **)&idx, sizeof(int)))
  *idx = 0;
  internal::parallel_filter(size, algorithm, idx, result_view, map_result_view);

  // copy back results
  internal::copy(result_buffer, *result, vecmem::copy::type::device_to_host);
  result->resize(*idx);

  // release the memory allocated for the index
  CHECK_ERROR(cudaFree(idx))

  return *result;
}

template <class Algorithm, typename R, typename T, typename... Arguments>
requires vecpar::algorithm::is_map_filter<Algorithm, R, T, Arguments...> R &
parallel_map_filter(Algorithm algorithm, vecmem::host_memory_resource &mr,
                    T &data, Arguments &...args) {
  return parallel_map_filter(algorithm, mr, cuda::getDefaultConfig(data.size()),
                             data, args...);
}
/*
template <class Algorithm, typename R, typename T, typename... Arguments>
requires vecpar::algorithm::is_mmap_filter<Algorithm, R, Arguments...> R &
parallel_map_filter(Algorithm algorithm, vecmem::host_memory_resource &mr,
                    vecpar::config config, T &data, Arguments&... args) {
  size_t size = data.size();
  // copy input data from host to device
  auto data_buffer = internal::copy.to(vecmem::get_data(data), internal::d_mem,
                                       vecmem::copy::type::host_to_device);
  auto data_view = vecmem::get_data(data_buffer);
  auto input = get_view_of_copied_container_or_obj(args...);

  auto fn = [&]<typename... P>(P&... params) {
        return internal::parallel_mmap<Algorithm, R, Arguments...>(
                config, data.size(), algorithm, data_view, params...);
  };
  std::apply(fn, input);

  // allocate result on host and device
  R *result = new R(size, &mr);
  auto result_buffer =
      internal::copy.to(vecmem::get_data(*result), internal::d_mem,
                        vecmem::copy::type::host_to_device);
  auto result_view = vecmem::get_data(result_buffer);

  int *idx; // global index
  CHECK_ERROR(cudaMallocManaged((void **)&idx, sizeof(int)))
  *idx = 0;
  internal::parallel_filter(size, algorithm, idx, result_view, data_view);

  // copy back results
  internal::copy(result_buffer, *result, vecmem::copy::type::device_to_host);
  result->resize(*idx);

  // release the memory allocated for the index
  CHECK_ERROR(cudaFree(idx))
  return *result;
}
*/

template <class Algorithm, typename R, typename T, typename... Arguments>
requires vecpar::algorithm::is_mmap_filter_1<Algorithm, T, Arguments...> R &
parallel_map_filter(Algorithm algorithm, vecmem::host_memory_resource &mr,
                    vecpar::config config, T &in_1, Arguments &...args) {
  size_t size = in_1.size();
  // copy input data from host to device
  auto buff_1 = internal::copy.to(vecmem::get_data(in_1), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_1 = vecmem::get_data(buff_1);

  internal::parallel_mmap<Algorithm, T, Arguments...>(
      config, in_1.size(), algorithm, data_view_1, args...);

  // allocate result on host and device
  R *result = new R(size, &mr);
  auto result_buffer =
      internal::copy.to(vecmem::get_data(*result), internal::d_mem,
                        vecmem::copy::type::host_to_device);
  auto result_view = vecmem::get_data(result_buffer);

  int *idx; // global index
  CHECK_ERROR(cudaMallocManaged((void **)&idx, sizeof(int)))
  *idx = 0;
  internal::parallel_filter(size, algorithm, idx, result_view, data_view_1);

  // copy back results
  internal::copy(result_buffer, *result, vecmem::copy::type::device_to_host);
  result->resize(*idx);

  // release the memory allocated for the index
  CHECK_ERROR(cudaFree(idx))

  return *result;
}
template <class Algorithm, typename R, typename T1, typename T2,
          typename... Arguments>
requires vecpar::algorithm::is_mmap_filter_2<Algorithm, T1, T2, Arguments...>
    R &parallel_map_filter(Algorithm algorithm,
                           vecmem::host_memory_resource &mr,
                           vecpar::config config, T1 &in_1, T2 &in_2,
                           Arguments &...args) {
  size_t size = in_1.size();
  // copy input data from host to device
  auto buff_1 = internal::copy.to(vecmem::get_data(in_1), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_1 = vecmem::get_data(buff_1);

  auto buff_2 = internal::copy.to(vecmem::get_data(in_2), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_2 = vecmem::get_data(buff_2);

  internal::parallel_mmap<Algorithm, T1, T2, Arguments...>(
      config, in_1.size(), algorithm, data_view_1, data_view_2, args...);

  // allocate result on host and device
  R *result = new R(size, &mr);
  auto result_buffer =
      internal::copy.to(vecmem::get_data(*result), internal::d_mem,
                        vecmem::copy::type::host_to_device);
  auto result_view = vecmem::get_data(result_buffer);

  int *idx; // global index
  CHECK_ERROR(cudaMallocManaged((void **)&idx, sizeof(int)))
  *idx = 0;
  internal::parallel_filter(size, algorithm, idx, result_view, data_view_1);

  // copy back results
  internal::copy(result_buffer, *result, vecmem::copy::type::device_to_host);
  result->resize(*idx);

  // release the memory allocated for the index
  CHECK_ERROR(cudaFree(idx))

  return *result;
}

template <class Algorithm, typename R, typename T1, typename T2, typename T3,
          typename... Arguments>
requires vecpar::algorithm::is_mmap_filter_3<Algorithm, T1, T2, T3,
                                             Arguments...>
    R &parallel_map_filter(Algorithm algorithm,
                           vecmem::host_memory_resource &mr,
                           vecpar::config config, T1 &in_1, T2 &in_2, T3 &in_3,
                           Arguments &...args) {
  size_t size = in_1.size();
  // copy input data from host to device
  auto buff_1 = internal::copy.to(vecmem::get_data(in_1), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_1 = vecmem::get_data(buff_1);

  auto buff_2 = internal::copy.to(vecmem::get_data(in_2), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_2 = vecmem::get_data(buff_2);

  auto buff_3 = internal::copy.to(vecmem::get_data(in_3), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_3 = vecmem::get_data(buff_3);

  internal::parallel_mmap<Algorithm, T1, T2, T3, Arguments...>(
      config, in_1.size(), algorithm, data_view_1, data_view_2, data_view_3,
      args...);

  // allocate result on host and device
  R *result = new R(size, &mr);
  auto result_buffer =
      internal::copy.to(vecmem::get_data(*result), internal::d_mem,
                        vecmem::copy::type::host_to_device);
  auto result_view = vecmem::get_data(result_buffer);

  int *idx; // global index
  CHECK_ERROR(cudaMallocManaged((void **)&idx, sizeof(int)))
  *idx = 0;
  internal::parallel_filter(size, algorithm, idx, result_view, data_view_1);

  // copy back results
  internal::copy(result_buffer, *result, vecmem::copy::type::device_to_host);
  result->resize(*idx);

  // release the memory allocated for the index
  CHECK_ERROR(cudaFree(idx))

  return *result;
}

template <class Algorithm, typename R, typename T1, typename T2, typename T3,
          typename T4, typename... Arguments>
requires vecpar::algorithm::is_mmap_filter_4<Algorithm, T1, T2, T3, T4,
                                             Arguments...>
    R &parallel_map_filter(Algorithm algorithm,
                           vecmem::host_memory_resource &mr,
                           vecpar::config config, T1 &in_1, T2 &in_2, T3 &in_3,
                           T4 &in_4, Arguments &...args) {
  size_t size = in_1.size();
  // copy input data from host to device
  auto buff_1 = internal::copy.to(vecmem::get_data(in_1), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_1 = vecmem::get_data(buff_1);

  auto buff_2 = internal::copy.to(vecmem::get_data(in_2), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_2 = vecmem::get_data(buff_2);

  auto buff_3 = internal::copy.to(vecmem::get_data(in_3), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_3 = vecmem::get_data(buff_3);

  auto buff_4 = internal::copy.to(vecmem::get_data(in_4), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_4 = vecmem::get_data(buff_4);

  internal::parallel_mmap<Algorithm, T1, T2, T3, T4, Arguments...>(
      config, in_1.size(), algorithm, data_view_1, data_view_2, data_view_3,
      data_view_4, args...);

  // allocate result on host and device
  R *result = new R(size, &mr);
  auto result_buffer =
      internal::copy.to(vecmem::get_data(*result), internal::d_mem,
                        vecmem::copy::type::host_to_device);
  auto result_view = vecmem::get_data(result_buffer);

  int *idx; // global index
  CHECK_ERROR(cudaMallocManaged((void **)&idx, sizeof(int)))
  *idx = 0;
  internal::parallel_filter(size, algorithm, idx, result_view, data_view_1);

  // copy back results
  internal::copy(result_buffer, *result, vecmem::copy::type::device_to_host);
  result->resize(*idx);

  // release the memory allocated for the index
  CHECK_ERROR(cudaFree(idx))

  return *result;
}

template <class Algorithm, typename R, typename T1, typename T2, typename T3,
          typename T4, typename T5, typename... Arguments>
requires vecpar::algorithm::is_mmap_filter_5<Algorithm, T1, T2, T3, T4, T5,
                                             Arguments...>
    R &parallel_map_filter(Algorithm algorithm,
                           vecmem::host_memory_resource &mr,
                           vecpar::config config, T1 &in_1, T2 &in_2, T3 &in_3,
                           T4 &in_4, T5 &in_5, Arguments &...args) {
  size_t size = in_1.size();
  // copy input data from host to device
  auto buff_1 = internal::copy.to(vecmem::get_data(in_1), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_1 = vecmem::get_data(buff_1);

  auto buff_2 = internal::copy.to(vecmem::get_data(in_2), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_2 = vecmem::get_data(buff_2);

  auto buff_3 = internal::copy.to(vecmem::get_data(in_3), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_3 = vecmem::get_data(buff_3);

  auto buff_4 = internal::copy.to(vecmem::get_data(in_4), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_4 = vecmem::get_data(buff_4);

  auto buff_5 = internal::copy.to(vecmem::get_data(in_5), internal::d_mem,
                                  vecmem::copy::type::host_to_device);
  auto data_view_5 = vecmem::get_data(buff_5);

  internal::parallel_mmap<Algorithm, T1, T2, T3, T4, T5, Arguments...>(
      config, in_1.size(), algorithm, data_view_1, data_view_2, data_view_3,
      data_view_4, data_view_5, args...);

  // allocate result on host and device
  R *result = new R(size, &mr);
  auto result_buffer =
      internal::copy.to(vecmem::get_data(*result), internal::d_mem,
                        vecmem::copy::type::host_to_device);
  auto result_view = vecmem::get_data(result_buffer);

  int *idx; // global index
  CHECK_ERROR(cudaMallocManaged((void **)&idx, sizeof(int)))
  *idx = 0;
  internal::parallel_filter(size, algorithm, idx, result_view, data_view_1);

  // copy back results
  internal::copy(result_buffer, *result, vecmem::copy::type::device_to_host);
  result->resize(*idx);

  // release the memory allocated for the index
  CHECK_ERROR(cudaFree(idx))

  return *result;
}

template <class Algorithm, typename R, typename T, typename... Arguments>
requires vecpar::algorithm::is_mmap_filter<Algorithm, R, Arguments...> R &
parallel_map_filter(Algorithm algorithm, vecmem::host_memory_resource &mr,
                    T &data, Arguments &...args) {

  return parallel_map_filter(algorithm, mr, cuda::getDefaultConfig(data.size()),
                             data, args...);
}
} // namespace vecpar::cuda
#endif // VECPAR_CUDA_HOSTMEM_HPP
