#ifndef VECPAR_DEVICE_MEMORY_HPP
#define VECPAR_DEVICE_MEMORY_HPP

#include <vecmem/containers/data/vector_view.hpp>
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/containers/data/vector_view.hpp>
#include <vecmem/utils/cuda/copy.hpp>

#include "internal.hpp"

namespace vecpar::cuda {

    template<typename Algorithm,
            class R = typename Algorithm::result_t,
            typename T,
            typename... Arguments,
            typename std::enable_if<!std::is_same<T, R>::value, void>::type* = nullptr>
    vecmem::data::vector_view<R>& parallel_map(Algorithm algorithm,
                                               vecpar::config config,
                                               vecmem::data::vector_view<R>& result_view,
                                               vecmem::data::vector_view<T>& data_view,
                                               Arguments... args) {
        internal::parallel_map(config,
                               data_view.size(),
                               algorithm,
                               result_view,
                               data_view,
                               args...);
       return result_view;
    }

    template<typename Algorithm,
            class R = typename Algorithm::result_t,
            typename T,
            typename... Arguments,
            typename std::enable_if<std::is_same<T, R>::value, void>::type* = nullptr>
    vecmem::data::vector_view<T>& parallel_map(Algorithm algorithm,
                                                 vecpar::config config,
                                                 vecmem::data::vector_view<T>& data_view,
                                                 Arguments... args) {
        internal::parallel_map(config,
                               data_view.size(),
                               algorithm,
                               data_view,
                               args...);
        return data_view;
    }


    template<typename Algorithm, typename T>
    vecmem::data::vector_view<T>& parallel_filter(Algorithm algorithm,
                                               vecpar::config config,
                                               vecmem::data::vector_view<T>& data_view) {

        internal::parallel_filter(config,
                                data_view.size(),
                                algorithm,
                                data_view);
        return data_view;
    }

    template<typename Algorithm, typename R>
    R& parallel_reduce(Algorithm algorithm,
                       vecpar::config config,
                       vecmem::data::vector_view<R>& data_view) {

        R* d_result;
        cudaMallocManaged(&d_result, sizeof(R));
        memset(d_result, 0, sizeof(R));

        internal::parallel_reduce(data_view.size(),
                                  algorithm,
                                  d_result,
                                  data_view);
        return *d_result;
    }


    template<class Algorithm,
            class R = typename Algorithm::result_t,
            class T, typename... Arguments,
            typename std::enable_if_t<std::is_base_of<vecpar::algorithm::parallelizable_map<R, T, Arguments...>, Algorithm>::value ||
                                      std::is_base_of<vecpar::algorithm::parallelizable_mmap<T, Arguments...>, Algorithm>::value, bool> = true>
    vecmem::data::vector_view<R>& parallel_algorithm(Algorithm algorithm,
                                          vecpar::config config,
                                          vecmem::data::vector_view<T>& data,
                                          Arguments... args) {

        return parallel_map<Algorithm, R, T, Arguments...>(algorithm, config, data, args...);
    }


/*
    template <typename INPUT_TYPE, typename Algorithm>
    void chain(INPUT_TYPE input, std::vector<Algorithm> algorithms) {

       // const auto composition = compose(algorithms...);

    }

*/

}
#endif //VECPAR_DEVICE_MEMORY_HPP
