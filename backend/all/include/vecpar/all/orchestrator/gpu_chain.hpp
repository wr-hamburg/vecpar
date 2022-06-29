#ifndef VECPAR_GPU_CHAIN_HPP
#define VECPAR_GPU_CHAIN_HPP

#include <cstdlib>
#include <vecmem/memory/host_memory_resource.hpp>

#include "default_chain.hpp"
#include "vecpar/cuda/detail/generic/raw_data.hpp"

/// different variants for chains when the code is built for
/// GPU CUDA and the data is initially in host memory
namespace vecpar {

///  1. Specialization for host_memory & result as vecmem::vector
template <typename Ri, typename Ti, typename... OtherInput>
class chain<vecmem::host_memory_resource, vecmem::vector<Ri>, vecmem::vector<Ti>,
            OtherInput...> {

public:
  chain(vecmem::host_memory_resource &mem) : m_mr(mem) {}

  template <typename First, typename... Rest>
  chain &with_algorithms(First first_alg, Rest... rest_alg) {
    /// cannot call with_algorithms more than once
    assertm(!algorithms_set, ALGORITHMS_ALREADY_SET);

    composition = compose(wrapper_first(first_alg), wrapper(rest_alg)...);
    algorithms_set = true;

    return *this;
  }

  vecmem::vector<Ri> execute(vecmem::vector<Ti> &coll, OtherInput... rest) {

    DEBUG_ACTION(printf("[GPU CHAIN EXECUTOR]\n");)

    /// cannot invoke chain execution without providing algorithms
    assertm(algorithms_set, MISSING_ALGORITHMS);

    /// run the chain
    cuda_data<Ri> d_result = composition(coll, rest...);

    /// copy result from device to host
    Ri *result = (Ri *)malloc(d_result.size * sizeof(Ri));
    CHECK_ERROR(cudaMemcpy(result, d_result.ptr, d_result.size * sizeof(Ri),
                           cudaMemcpyDeviceToHost));

    /// release the device memory
    CHECK_ERROR(cudaFree(d_result.ptr))

    /// convert to vecmem::vector<Ri>
    vecmem::vector<Ri> desired_result(d_result.size, &m_mr);
    desired_result.assign(result, result + d_result.size);

    return desired_result;
  }

private:
  template <class Algorithm,
            class input_t = typename Algorithm::input_t,
            class result_t = typename Algorithm::result_t,
            class input_ti = typename Algorithm::input_ti,
            class result_ti = typename Algorithm::result_ti>

  auto wrapper_first(Algorithm &algorithm) {
    return [&](vecmem::vector<input_ti> &coll, OtherInput... otherInput) {
      size_t size = coll.size();

      /// copy the initial collection to the device
      Ti *d_data = NULL;
      CHECK_ERROR(cudaMalloc((void **)&d_data, size * sizeof(Ti)))
      CHECK_ERROR(cudaMemcpy(d_data, coll.data(), size * sizeof(Ti),
                             cudaMemcpyHostToDevice));

      /// convert into raw pointer & size
      cuda_data<Ti> input{d_data, size};

      if constexpr (std::is_base_of<
                        vecpar::algorithm::parallelizable_mmap_1<result_t, OtherInput...>,
                        Algorithm>::value) {
        /// make sure the input host vector is also changed
        cuda_data<result_ti> partial_result = vecpar::cuda_raw::parallel_algorithm<Algorithm, result_t, input_t>(algorithm, m_config,
                                                                                                     input, otherInput...);
        return vecpar::cuda_raw::copy_intermediate_result<Algorithm, result_ti>(
            coll, partial_result);
      } else if constexpr (std::is_base_of<
                               vecpar::algorithm::parallelizable_map_1<result_t, input_t,
                                                            OtherInput...>,
                               Algorithm>::value) {
        return vecpar::cuda_raw::parallel_algorithm(algorithm, m_config, input,
                                                    otherInput...);
      } else {
        return vecpar::cuda_raw::parallel_algorithm(algorithm, m_config, input);
      }
    };
  }

  template <class Algorithm,
            class input_ti = typename Algorithm::input_ti,
            class result_ti = typename Algorithm::result_ti>

  auto wrapper(Algorithm &algorithm) {
    return [&](cuda_data<input_ti> input) {
      return vecpar::cuda_raw::parallel_algorithm(algorithm, m_config, input);
    };
  }

private:
  vecmem::host_memory_resource &m_mr;
  vecpar::config m_config;
  std::function<cuda_data<Ri>(vecmem::vector<Ti> &, OtherInput...)> composition;
  bool algorithms_set = false;
};

///  Specialization for host_memory & result as an object R
template <typename R, typename Ti, typename... OtherInput>
class chain<vecmem::host_memory_resource, R, vecmem::vector<Ti>, OtherInput...> {

public:
  chain(vecmem::host_memory_resource &mem) : m_mr(mem) {}

  template <typename First, typename... Rest>
  chain &with_algorithms(First first_alg, Rest... rest_alg) {
    /// cannot call with_algorithms more than once
    assertm(!algorithms_set, ALGORITHMS_ALREADY_SET);

    composition = compose(wrapper_first(first_alg), wrapper(rest_alg)...);
    algorithms_set = true;

    return *this;
  }

  R execute(vecmem::vector<Ti> &coll, OtherInput... rest) {
    DEBUG_ACTION(printf("[GPU CHAIN EXECUTOR]\n");)

    /// cannot invoke chain execution without providing algorithms
    assertm(algorithms_set, MISSING_ALGORITHMS);

    /// run the chain
    cuda_data<R> d_result = composition(coll, rest...);

    /// copy result from device to host
    R *result = (R *)malloc(sizeof(R));
    CHECK_ERROR(
        cudaMemcpy(result, d_result.ptr, sizeof(R), cudaMemcpyDeviceToHost));
    CHECK_ERROR(cudaFree(d_result.ptr))

    return *result;
  }

private:
  template <class Algorithm,
            class input_t = typename Algorithm::input_t,
            class result_t = typename Algorithm::result_t,
            class input_ti = typename Algorithm::input_ti,
            class result_ti = typename Algorithm::result_ti>

  auto wrapper_first(Algorithm &algorithm) {
    return [&](vecmem::vector<input_ti> &coll, OtherInput... otherInput) {
      size_t size = coll.size();

      /// copy the initial collection to the device
      Ti *d_data = NULL;
      CHECK_ERROR(cudaMalloc((void **)&d_data, size * sizeof(Ti)))
      CHECK_ERROR(cudaMemcpy(d_data, coll.data(), size * sizeof(Ti),
                             cudaMemcpyHostToDevice));

      /// convert into raw pointer & size
      cuda_data<Ti> input{d_data, size};

      if constexpr (std::is_base_of<
                        vecpar::detail::parallel_mmap_1<result_t, OtherInput...>,
                        Algorithm>::value) {
        /// make sure the input host vector is also changed
        return vecpar::cuda_raw::copy_intermediate_result<Algorithm, result_ti>(
            coll, vecpar::cuda_raw::parallel_algorithm(algorithm, m_config,
                                                       input, otherInput...));
      } else if constexpr (std::is_base_of<
                               vecpar::detail::parallel_map_1<result_t, input_t,
                                                            OtherInput...>,
                               Algorithm>::value) {
        return vecpar::cuda_raw::parallel_algorithm<Algorithm, result_t, input_t, OtherInput...>(algorithm, m_config, input,
                                                    otherInput...);
      } else {
        return vecpar::cuda_raw::parallel_algorithm<Algorithm, result_t, input_t>(
            algorithm, m_config, input);
      }
    };
  }

  template <class Algorithm,
            class input_t = typename Algorithm::input_t,
            class result_t = typename Algorithm::result_t,
            class input_ti = typename Algorithm::input_ti,
            class result_ti = typename Algorithm::result_ti>

  auto wrapper(Algorithm &algorithm) {
    return [&](cuda_data<input_ti> input) {
      return vecpar::cuda_raw::parallel_algorithm<Algorithm> (algorithm, m_config, input);
    };
  }

private:
  vecmem::host_memory_resource &m_mr;
  vecpar::config m_config;
  std::function<cuda_data<R>(vecmem::vector<Ti> &, OtherInput...)> composition;
  bool algorithms_set = false;
};
} // namespace vecpar
#endif // VECPAR_GPU_CHAIN_HPP
