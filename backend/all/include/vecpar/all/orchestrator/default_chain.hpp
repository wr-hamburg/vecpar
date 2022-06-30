#ifndef VECPAR_DEFAULT_CHAIN_HPP
#define VECPAR_DEFAULT_CHAIN_HPP

#include "common.hpp"
#include <vecmem/containers/vector.hpp>
#include <vecmem/memory/host_memory_resource.hpp>

namespace vecpar {

template <class MemoryResource, typename R, typename T, typename... OtherInput>
class chain {

public:
  chain(MemoryResource &mem) : m_mr(mem) {}

  chain &with_config(vecpar::config config) {
    m_config = config;
    return *this;
  }

  template <typename... Args> chain &with_algorithms(Args... args) {
    /// cannot call with_algorithms more than once
    assertm(!algorithms_set, ALGORITHMS_ALREADY_SET);

    composition = compose(wrapper(args)...);
    algorithms_set = true;

    return *this;
  }

  R execute(T &coll, OtherInput... rest) {

    DEBUG_ACTION(printf("[DEFAULT CHAIN EXECUTOR]\n");)

    /// cannot invoke chain execution without providing algorithms
    assertm(algorithms_set, MISSING_ALGORITHMS);

    return composition(coll, rest...);
  }

private:
  template <class Algorithm, class input_t = typename Algorithm::input_t,
            class result_t = typename Algorithm::result_t>
  auto wrapper(Algorithm &algorithm) {
    return [&](input_t &coll, OtherInput... otherInput) -> result_t & {
      if constexpr (vecpar::algorithm::is_map<Algorithm, result_t, input_t,
                                              OtherInput...> ||
                    vecpar::algorithm::is_mmap<Algorithm, result_t,
                                               OtherInput...>) {
        return vecpar::parallel_algorithm(algorithm, m_mr, m_config, coll,
                                          otherInput...);
      } else {
        return vecpar::parallel_algorithm(algorithm, m_mr, m_config, coll);
      }
    };
  }

protected:
  MemoryResource &m_mr;
  vecpar::config m_config;
  std::function<R(T &, OtherInput...)> composition;
  bool algorithms_set = false;
};
} // namespace vecpar

#endif // VECPAR_DEFAULT_CHAIN_HPP