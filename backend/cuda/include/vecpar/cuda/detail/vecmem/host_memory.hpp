#ifndef VECPAR_CUDA_HOSTMEM_HPP
#define VECPAR_CUDA_HOSTMEM_HPP

#include "vecpar/cuda/detail/common/cuda_utils.hpp"
#include <vecmem/containers/data/vector_view.hpp>
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/cuda/managed_memory_resource.hpp>
#include <vecmem/memory/host_memory_resource.hpp>

#include "internal.hpp"

namespace vecpar::cuda {

template <typename... T>
std::tuple<std::conditional_t<
    (std::is_object<T>::value && Jagged_vector_type<T>),
    vecmem::data::jagged_vector_data<value_type_t<T>>,
    std::conditional_t<(std::is_object<T>::value && Vector_type<T>),
                       vecmem::data::vector_buffer<value_type_t<T>>, T>>...>
get_buffer_of_copied_container_or_obj(T &...obj) {
  return {([](T &i) {
    if constexpr (Jagged_vector_type<T>) {
      auto data = vecmem::get_data(i);
      return data;
    } else if constexpr (Vector_type<T>) {
      auto buffer = internal::copy.to(vecmem::get_data(i), internal::d_mem,
                                      vecmem::copy::type::host_to_device);
      return buffer;
    } else {
      return i;
    }
  }(obj))...};
}

template <typename Algorithm,
          typename R = typename Algorithm::intermediate_result_t, typename T,
          typename... Arguments>
requires vecpar::detail::is_map<Algorithm, R, T, Arguments...> R &
parallel_map(Algorithm algorithm, vecmem::host_memory_resource &mr,
             vecpar::config config, T &data, Arguments &...args) {

  auto fn_jagged = [&]<typename... P>(P & ...obj)
                       ->std::tuple<std::conditional_t<
                           jagged_view<P>,
                           vecmem::data::jagged_vector_buffer<
                               typename extract_value_type<P>::value_type>,
                           P>...> {
    return {([&](P &i) {
      if constexpr (jagged_view<P>) {
        auto buffer =
            internal::copy.to(vecmem::get_data(i), internal::d_mem, &mr,
                              vecmem::copy::type::host_to_device);
        return buffer;
      } else {
        return std::move(i);
      }
    }(obj))...};
  };

  // allocate map result on host and copy to device
  R *map_result = new R(data.size(), &mr);

  auto result_buffer_b = get_buffer_of_copied_container_or_obj(*map_result);
  auto result_buffer = get<0>(std::apply(fn_jagged, result_buffer_b));
  auto result_view = vecmem::get_data(result_buffer);

  // input data
  auto input = get_buffer_of_copied_container_or_obj(data, args...);
  auto input_j = std::apply(fn_jagged, input);

  auto fn = [&]<typename... P>(P & ...params) {
    return internal::parallel_map<Algorithm, R, T, Arguments...>(
        config, data.size(), algorithm, result_view, params...);
  };

  std::apply(fn, input_j);

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
requires vecpar::detail::is_mmap<Algorithm, R, Arguments...> R &
parallel_map(Algorithm algorithm, vecmem::host_memory_resource &mr,
             vecpar::config config, T &data, Arguments &...args) {

  auto fn_jagged =
      [&]<typename... P>(P & ...obj)
          ->std::tuple<std::conditional_t<
              (std::is_same_v<P,
                              vecmem::data::jagged_vector_data<
                                  typename extract_value_type<P>::value_type>>),
              vecmem::data::jagged_vector_buffer<
                  typename extract_value_type<P>::value_type>,
              P>...> {
    return {([&](P &i) {
      if constexpr (std::is_same_v<
                        P, vecmem::data::jagged_vector_data<
                               typename extract_value_type<P>::value_type>>) {
        auto buffer =
            internal::copy.to(vecmem::get_data(i), internal::d_mem, &mr,
                              vecmem::copy::type::host_to_device);
        return buffer;
      } else {
        return std::move(i);
      }
    }(obj))...};
  };
  // copy input data from host to device
  auto data_b = get_buffer_of_copied_container_or_obj(data);
  auto data_buffer = get<0>(std::apply(fn_jagged, data_b));
  auto data_view = vecmem::get_data(data_buffer);

  auto input = get_buffer_of_copied_container_or_obj(args...);
  auto input_j = std::apply(fn_jagged, input);

  auto fn = [&]<typename... P>(P & ...params) {
    return internal::parallel_mmap<Algorithm, R, Arguments...>(
        config, data.size(), algorithm, data_view, params...);
  };

  std::apply(fn, input_j);

  internal::copy(data_buffer, data, vecmem::copy::type::device_to_host);
  return data;
}

template <typename Algorithm,
          typename R = typename Algorithm::intermediate_result_t, typename T,
          typename... Arguments>
requires vecpar::detail::is_mmap<Algorithm, R, Arguments...> R &
parallel_map(Algorithm algorithm,
             __attribute__((unused)) vecmem::host_memory_resource &mr, T &data,
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

  internal::parallel_reduce<Algorithm, typename R::value_type>(data.size(), algorithm, d_result, data_view);
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

template <class Algorithm, typename Result = typename Algorithm::result_t,
          typename R = typename Algorithm::intermediate_result_t, typename T,
          typename... Arguments>
requires vecpar::algorithm::is_map_reduce<Algorithm, Result, R, T, Arguments...>
    Result &
    parallel_map_reduce(Algorithm algorithm, vecmem::host_memory_resource &mr,
                        vecpar::config config, T &data, Arguments &...args) {

  auto fn_jagged = [&]<typename... P>(P & ...obj)
                       ->std::tuple<std::conditional_t<
                           jagged_view<P>,
                           vecmem::data::jagged_vector_buffer<
                               typename extract_value_type<P>::value_type>,
                           P>...> {
    return {([&](P &i) {
      if constexpr (jagged_view<P>) {
        auto buffer =
            internal::copy.to(vecmem::get_data(i), internal::d_mem, &mr,
                              vecmem::copy::type::host_to_device);
        return buffer;
      } else {
        return std::move(i);
      }
    }(obj))...};
  };

  // allocate temp map result on host and copy to device
  R *map_result = new R(data.size(), &mr);
  auto result_buffer_b = get_buffer_of_copied_container_or_obj(*map_result);
  auto result_buffer = get<0>(std::apply(fn_jagged, result_buffer_b));
  auto result_view = vecmem::get_data(result_buffer);

  // input data
  auto input = get_buffer_of_copied_container_or_obj(data, args...);
  auto input_j = std::apply(fn_jagged, input);

  auto fn = [&]<typename... P>(P & ...params) {
    return internal::parallel_map<Algorithm, R, T, Arguments...>(
        config, data.size(), algorithm, result_view, params...);
  };

  std::apply(fn, input_j);

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

template <class Algorithm, typename Result = typename Algorithm::result_t,
          typename R = typename Algorithm::intermediate_result_t, typename T,
          typename... Arguments>
requires vecpar::algorithm::is_mmap_reduce<Algorithm, Result, R, Arguments...>
    Result &parallel_map_reduce(Algorithm algorithm,
                                __attribute__((unused))
                                vecmem::host_memory_resource &mr,
                                vecpar::config config, T &data,
                                Arguments &...args) {
  auto fn_jagged =
      [&]<typename... P>(P & ...obj)
          ->std::tuple<std::conditional_t<
              (std::is_same_v<P,
                              vecmem::data::jagged_vector_data<
                                  typename extract_value_type<P>::value_type>>),
              vecmem::data::jagged_vector_buffer<
                  typename extract_value_type<P>::value_type>,
              P>...> {
    return {([&](P &i) {
      if constexpr (std::is_same_v<
                        P, vecmem::data::jagged_vector_data<
                               typename extract_value_type<P>::value_type>>) {
        auto buffer =
            internal::copy.to(vecmem::get_data(i), internal::d_mem, &mr,
                              vecmem::copy::type::host_to_device);
        return buffer;
      } else {
        return std::move(i);
      }
    }(obj))...};
  };

  // copy input data from host to device
  // copy input data from host to device
  auto data_b = get_buffer_of_copied_container_or_obj(data);
  auto data_buffer = get<0>(std::apply(fn_jagged, data_b));
  auto data_view = vecmem::get_data(data_buffer);

  auto input = get_buffer_of_copied_container_or_obj(args...);
  auto input_j = std::apply(fn_jagged, input);

  auto fn = [&]<typename... P>(P & ...params) {
    return internal::parallel_mmap<Algorithm, R, Arguments...>(
        config, data.size(), algorithm, data_view, params...);
  };

  std::apply(fn, input_j);

  Result *result = (Result *)malloc(sizeof(Result));
  Result *d_result;

  CHECK_ERROR(cudaMalloc((void **)&d_result, sizeof(Result)))
  CHECK_ERROR(cudaMemset(d_result, 0, sizeof(Result)))

  internal::parallel_reduce<Algorithm, Result>(data.size(), algorithm, d_result, data_view);

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

template <class Algorithm, typename R, typename T, typename... Arguments>
requires vecpar::algorithm::is_map_filter<Algorithm, R, T, Arguments...> R &
parallel_map_filter(Algorithm algorithm, vecmem::host_memory_resource &mr,
                    vecpar::config config, T &data, Arguments &...args) {
  size_t size = data.size();
  auto fn_jagged = [&]<typename... P>(P & ...obj)
                       ->std::tuple<std::conditional_t<
                           jagged_view<P>,
                           vecmem::data::jagged_vector_buffer<
                               typename extract_value_type<P>::value_type>,
                           P>...> {
    return {([&](P &i) {
      if constexpr (jagged_view<P>) {
        auto buffer =
            internal::copy.to(vecmem::get_data(i), internal::d_mem, &mr,
                              vecmem::copy::type::host_to_device);
        return buffer;
      } else {
        return std::move(i);
      }
    }(obj))...};
  };

  // allocate map result on host and copy to device
  R *map_result = new R(data.size(), &mr);

  auto result_buffer_b = get_buffer_of_copied_container_or_obj(*map_result);
  auto result_buffer_m = get<0>(std::apply(fn_jagged, result_buffer_b));
  auto result_view_m = vecmem::get_data(result_buffer_m);

  // input data
  auto input = get_buffer_of_copied_container_or_obj(data, args...);
  auto input_j = std::apply(fn_jagged, input);

  auto fn = [&]<typename... P>(P & ...params) {
    return internal::parallel_map<Algorithm, R, T, Arguments...>(
        config, data.size(), algorithm, result_view_m, params...);
  };
  std::apply(fn, input_j);

  // allocate result on host and device
  R *result = new R(size, &mr);
  auto result_buffer =
      internal::copy.to(vecmem::get_data(*result), internal::d_mem,
                        vecmem::copy::type::host_to_device);
  auto result_view = vecmem::get_data(result_buffer);

  int *idx; // global index
  CHECK_ERROR(cudaMallocManaged((void **)&idx, sizeof(int)))
  *idx = 0;
  internal::parallel_filter(size, algorithm, idx, result_view, result_view_m);

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

template <class Algorithm, typename R, typename T, typename... Arguments>
requires vecpar::algorithm::is_mmap_filter<Algorithm, R, Arguments...> R &
parallel_map_filter(Algorithm algorithm, vecmem::host_memory_resource &mr,
                    vecpar::config config, T &data, Arguments &...args) {
  size_t size = data.size();

  auto fn_jagged =
      [&]<typename... P>(P & ...obj)
          ->std::tuple<std::conditional_t<
              (std::is_same_v<P,
                              vecmem::data::jagged_vector_data<
                                  typename extract_value_type<P>::value_type>>),
              vecmem::data::jagged_vector_buffer<
                  typename extract_value_type<P>::value_type>,
              P>...> {
    return {([&](P &i) {
      if constexpr (std::is_same_v<
                        P, vecmem::data::jagged_vector_data<
                               typename extract_value_type<P>::value_type>>) {
        auto buffer =
            internal::copy.to(vecmem::get_data(i), internal::d_mem, &mr,
                              vecmem::copy::type::host_to_device);
        return buffer;
      } else {
        return std::move(i);
      }
    }(obj))...};
  };
  // copy input data from host to device
  auto data_b = get_buffer_of_copied_container_or_obj(data);
  auto data_buffer = get<0>(std::apply(fn_jagged, data_b));
  auto data_view = vecmem::get_data(data_buffer);

  auto input = get_buffer_of_copied_container_or_obj(args...);
  auto input_j = std::apply(fn_jagged, input);

  auto fn = [&]<typename... P>(P & ...params) {
    return internal::parallel_mmap<Algorithm, R, Arguments...>(
        config, data.size(), algorithm, data_view, params...);
  };
  std::apply(fn, input_j);

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
template <class Algorithm, typename R, typename T, typename... Arguments>
requires vecpar::algorithm::is_mmap_filter<Algorithm, R, Arguments...> R &
parallel_map_filter(Algorithm algorithm, vecmem::host_memory_resource &mr,
                    T &data, Arguments &...args) {

  return parallel_map_filter(algorithm, mr, cuda::getDefaultConfig(data.size()),
                             data, args...);
}
} // namespace vecpar::cuda
#endif // VECPAR_CUDA_HOSTMEM_HPP
