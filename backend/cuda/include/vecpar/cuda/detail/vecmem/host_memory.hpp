#ifndef VECPAR_CUDA_HOSTMEM_HPP
#define VECPAR_CUDA_HOSTMEM_HPP

#include <vecmem/containers/data/vector_view.hpp>
#include <vecmem/memory/cuda/managed_memory_resource.hpp>
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/host_memory_resource.hpp>
#include "vecpar/cuda/detail/common/cuda_utils.hpp"

#include "internal.hpp"

namespace vecpar::cuda {

    template<typename Algorithm,
            class R = typename Algorithm::result_t,
            typename T,
            typename... Arguments,
            typename std::enable_if<!std::is_same<T, R>::value, void>::type* = nullptr>
    vecmem::vector<R>& parallel_map(Algorithm algorithm,
                                    vecmem::host_memory_resource& mr,
                                    vecpar::config config,
                                    vecmem::vector<T>& data,
                                    Arguments... args) {

        // copy input data from host to device
        auto data_buffer = internal::copy.to(vecmem::get_data(data), internal::d_mem, vecmem::copy::type::host_to_device);
        auto data_view = vecmem::get_data(data_buffer);

        // allocate map result on host and copy to device
        vecmem::vector<R>* map_result = new vecmem::vector<R>(data.size(), &mr);
        auto result_buffer = internal::copy.to(vecmem::get_data(*map_result), internal::d_mem, vecmem::copy::type::host_to_device);
        auto result_view = vecmem::get_data(result_buffer);

        internal::parallel_map(config,
                               data.size(),
                               algorithm,
                               result_view,
                               data_view,
                               args...);

        internal::copy(result_buffer, *map_result, vecmem::copy::type::device_to_host);
        return *map_result;
    }

    template<typename Algorithm,
            class R = typename Algorithm::result_t,
            typename T,
            typename... Arguments,
            typename std::enable_if<!std::is_same<T, R>::value, void>::type* = nullptr>
    vecmem::vector<R>& parallel_map(Algorithm algorithm,
                                    vecmem::host_memory_resource& mr,
                                    vecmem::vector<T>& data,
                                    Arguments... args) {
        return parallel_map(algorithm, mr, cuda::getDefaultConfig(data.size()), data, args...);
    }


    template<typename Algorithm,
            class R = typename Algorithm::result_t,
            typename T,
            typename... Arguments,
            typename std::enable_if<std::is_same<T, R>::value, void>::type* = nullptr>
    vecmem::vector<R>& parallel_map(Algorithm algorithm,
                                    __attribute__((unused)) vecmem::host_memory_resource& mr,
                                    vecpar::config config,
                                    vecmem::vector<T>& data,
                                    Arguments... args) {

        // copy input data from host to device
        auto data_buffer = internal::copy.to(vecmem::get_data(data), internal::d_mem, vecmem::copy::type::host_to_device);
        auto data_view = vecmem::get_data(data_buffer);

        internal::parallel_map(config,
                               data.size(),
                               algorithm,
                               data_view,
                               args...);

        internal::copy(data_buffer, data, vecmem::copy::type::device_to_host);
        return data;
    }

    template<typename Algorithm,
            class R = typename Algorithm::result_t,
            typename T,
            typename... Arguments,
            typename std::enable_if<std::is_same<T, R>::value, void>::type* = nullptr>
    vecmem::vector<R>& parallel_map(Algorithm algorithm,
                                    __attribute__((unused)) vecmem::host_memory_resource& mr,
                                    vecmem::vector<T>& data,
                                    Arguments... args) {
        return parallel_map(algorithm, mr, cuda::getDefaultConfig(data.size()), data, args...);
    }

    template<typename Algorithm, typename R>
    R& parallel_reduce(Algorithm algorithm,
                       __attribute__((unused)) vecmem::host_memory_resource& mr,
                       vecmem::vector<R>& data) {

        // copy input data from host to device
        auto data_buffer = internal::copy.to(vecmem::get_data(data), internal::d_mem, vecmem::copy::type::host_to_device);
        auto data_view = vecmem::get_data(data_buffer);

        // TODO: return a pointer to host from here
        R* d_result;
        cudaMallocManaged(&d_result, sizeof(R));
        memset(d_result, 0, sizeof(R));

        internal::parallel_reduce(data.size(),
                                  algorithm,
                                  d_result,
                                  data_view);
        return *d_result;
    }

    template<typename Algorithm, typename R>
    vecmem::vector<R>& parallel_filter(Algorithm algorithm,
                                       vecmem::host_memory_resource& mr,
                                       vecmem::vector<R>& data) {

        // copy input data from host to device
        auto data_buffer = internal::copy.to(vecmem::get_data(data), internal::d_mem, vecmem::copy::type::host_to_device);
        auto data_view = vecmem::get_data(data_buffer);

        // allocate result on host and device
        vecmem::vector<R>* result = new vecmem::vector<R>(data.size(), &mr);
        auto result_buffer = internal::copy.to(vecmem::get_data(*result), internal::d_mem,
                                     vecmem::copy::type::host_to_device);
        auto result_view = vecmem::get_data(result_buffer);

        int* idx; // global index
        CHECK_ERROR(cudaMallocManaged((void**)&idx, sizeof(int)))
        *idx = 0;

        internal::parallel_filter(data.size(),
                                  algorithm,
                                  idx,
                                  result_view,
                                  data_view);

        //copy back results
        internal::copy(result_buffer, *result, vecmem::copy::type::device_to_host);
        result->resize(*idx);

        // release the memory allocated for the index
        CHECK_ERROR(cudaFree(idx))

        return *result;
    }

    template<class Algorithm,
            typename R,
            typename T,
            typename... Arguments,
            typename std::enable_if<!std::is_same<T, R>::value, void>::type* = nullptr>
    R& parallel_map_reduce(Algorithm algorithm,
                           vecmem::host_memory_resource& mr,
                           vecpar::config config,
                           vecmem::vector<T>& data,
                           Arguments... args)  {
        // copy input data from host to device
        auto data_buffer = internal::copy.to(vecmem::get_data(data), internal::d_mem, vecmem::copy::type::host_to_device);
        auto data_view = vecmem::get_data(data_buffer);

        // allocate temp map result on host and copy to device
        vecmem::vector<R> map_result(data.size(), &mr);
        auto result_buffer = internal::copy.to(vecmem::get_data(map_result), internal::d_mem, vecmem::copy::type::host_to_device);
        auto result_view = vecmem::get_data(result_buffer);

        internal::parallel_map(config,
                               data.size(),
                               algorithm,
                               result_view,
                               data_view,
                               args...);

        R* result = (R*) malloc(sizeof (R));
        R* d_result;

        CHECK_ERROR(cudaMalloc((void**)&d_result, sizeof(R)))
        CHECK_ERROR(cudaMemset(d_result, 0, sizeof(R)))

        internal::parallel_reduce(data.size(),
                                  algorithm,
                                  d_result,
                                  result_view);

        CHECK_ERROR(cudaMemcpy(result, d_result, sizeof(R), cudaMemcpyDeviceToHost))
        CHECK_ERROR(cudaFree(d_result))

        return *result;
    }

    template<class Algorithm,
            typename R,
            typename T,
            typename... Arguments,
            typename std::enable_if<!std::is_same<T, R>::value, void>::type* = nullptr>
    R& parallel_map_reduce(Algorithm algorithm,
                           vecmem::host_memory_resource& mr,
                           vecmem::vector<T>& data,
                           Arguments... args)  {
        return parallel_map_reduce(algorithm, mr, cuda::getDefaultConfig(data.size()), data, args...);
    }

    template<class Algorithm,
            typename R,
            typename T,
            typename... Arguments,
            typename std::enable_if<std::is_same<T, R>::value, void>::type* = nullptr>
    R& parallel_map_reduce(Algorithm algorithm,
                           __attribute__((unused)) vecmem::host_memory_resource& mr,
                           vecpar::config config,
                           vecmem::vector<T>& data,
                           Arguments... args)  {
        // copy input data from host to device
        auto data_buffer = internal::copy.to(vecmem::get_data(data), internal::d_mem, vecmem::copy::type::host_to_device);
        auto data_view = vecmem::get_data(data_buffer);

        // allocate temp map result on host and copy to device
  //      vecmem::vector<R> map_result(data.size(), &mr);
   //     auto result_buffer = copy.to(vecmem::get_data(map_result), d_mem, vecmem::copy::type::host_to_device);
    //    auto result_view = vecmem::get_data(result_buffer);

        internal::parallel_map(config,
                               data.size(),
                               algorithm,
                               data_view,
                               args...);

        R* result = (R*) malloc(sizeof (R));
        R* d_result;

        CHECK_ERROR(cudaMalloc((void**)&d_result, sizeof(R)))
        CHECK_ERROR(cudaMemset(d_result, 0, sizeof(R)))

        internal::parallel_reduce(data.size(),
                                  algorithm,
                                  d_result,
                                  data_view);

        CHECK_ERROR(cudaMemcpy(result, d_result, sizeof(R), cudaMemcpyDeviceToHost))
        CHECK_ERROR(cudaFree(d_result))

        return *result;
    }

    template<class Algorithm,
            typename R,
            typename T,
            typename... Arguments,
            typename std::enable_if<std::is_same<T, R>::value, void>::type* = nullptr>
    R& parallel_map_reduce(Algorithm algorithm,
                           __attribute__((unused)) vecmem::host_memory_resource& mr,
                           vecmem::vector<T>& data,
                           Arguments... args)  {
        return parallel_map_reduce(algorithm, mr, cuda::getDefaultConfig(data.size()), data, args...);
    }

    template<class Algorithm,
            typename R,
            typename T,
            typename... Arguments,
            typename std::enable_if<!std::is_same<T, R>::value, void>::type* = nullptr>
    vecmem::vector<R>& parallel_map_filter(Algorithm algorithm,
                                           vecmem::host_memory_resource& mr,
                                           vecpar::config config,
                                           vecmem::vector<T>& data,
                                           Arguments... args)  {
      size_t size = data.size();
      // copy input data from host to device
      auto data_buffer = internal::copy.to(vecmem::get_data(data), internal::d_mem,
                                 vecmem::copy::type::host_to_device);
      auto data_view = vecmem::get_data(data_buffer);

      // allocate temp map result on host and copy to device
      vecmem::vector<R> map_result(size, &mr);
      auto map_result_buffer = internal::copy.to(vecmem::get_data(map_result), internal::d_mem,
                                       vecmem::copy::type::host_to_device);
      auto map_result_view = vecmem::get_data(map_result_buffer);

      internal::parallel_map(config,
                             size,
                             algorithm,
                             map_result_view,
                             data_view,
                             args...);

      // allocate result on host and device
      vecmem::vector<R> *result = new vecmem::vector<R>(size, &mr);
      auto result_buffer = internal::copy.to(vecmem::get_data(*result), internal::d_mem,
                                   vecmem::copy::type::host_to_device);
      auto result_view = vecmem::get_data(result_buffer);

      int *idx; // global index
      CHECK_ERROR(cudaMallocManaged((void **)&idx, sizeof(int)))
      *idx = 0;
      internal::parallel_filter(size,
                                algorithm,
                                idx,
                                result_view,
                                map_result_view);

      // copy back results
      internal::copy(result_buffer, *result, vecmem::copy::type::device_to_host);
      result->resize(*idx);

      // release the memory allocated for the index
      CHECK_ERROR(cudaFree(idx))

      return *result;
    }

    template<class Algorithm,
            typename R,
            typename T,
            typename... Arguments,
            typename std::enable_if<!std::is_same<T, R>::value, void>::type* = nullptr>
    vecmem::vector<R>& parallel_map_filter(Algorithm algorithm,
                                           vecmem::host_memory_resource& mr,
                                           vecmem::vector<T>& data,
                                           Arguments... args)  {
        return parallel_map_filter(algorithm, mr, cuda::getDefaultConfig(data.size()), data, args...);
    }

    template<class Algorithm,
            typename R,
            typename T,
            typename... Arguments,
            typename std::enable_if<std::is_same<T, R>::value, void>::type* = nullptr>
    vecmem::vector<R>& parallel_map_filter(Algorithm algorithm,
                                           vecmem::host_memory_resource& mr,
                                           vecpar::config config,
                                           vecmem::vector<T>& data,
                                           Arguments... args)  {
      size_t size = data.size();
      // copy input data from host to device
      auto data_buffer = internal::copy.to(vecmem::get_data(data), internal::d_mem,
                                 vecmem::copy::type::host_to_device);
      auto data_view = vecmem::get_data(data_buffer);

      // allocate temp map result on host and copy to device
      //  vecmem::vector<R> map_result(data.size(), &mr);
      //  auto map_result_buffer = copy.to(vecmem::get_data(map_result), d_mem, vecmem::copy::type::host_to_device);
      //  auto map_result_view = vecmem::get_data(map_result_buffer);

        internal::parallel_map(config,
                               size,
                               algorithm,
                               data_view,
                               args...);

        // allocate result on host and device
        vecmem::vector<R>* result = new vecmem::vector<R>(size, &mr);
        auto result_buffer = internal::copy.to(vecmem::get_data(*result), internal::d_mem,
                                     vecmem::copy::type::host_to_device);
        auto result_view = vecmem::get_data(result_buffer);

        int* idx; // global index
        CHECK_ERROR(cudaMallocManaged((void**)&idx, sizeof(int)))
        *idx = 0;
        internal::parallel_filter(size,
                                  algorithm,
                                  idx,
                                  result_view,
                                  data_view);

        //copy back results
        internal::copy(result_buffer, *result, vecmem::copy::type::device_to_host);
        result->resize(*idx);

        // release the memory allocated for the index
        CHECK_ERROR(cudaFree(idx))
        return *result;
    }

    template<class Algorithm,
            typename R,
            typename T,
            typename... Arguments,
            typename std::enable_if<std::is_same<T, R>::value, void>::type* = nullptr>
    vecmem::vector<R>& parallel_map_filter(Algorithm algorithm,
                                           vecmem::host_memory_resource& mr,
                                           vecmem::vector<T>& data,
                                           Arguments... args)  {

        return parallel_map_filter(algorithm, mr, cuda::getDefaultConfig(data.size()), data, args...);
    }
}
#endif //VECPAR_CUDA_HOSTMEM_HPP
