#ifndef VECPAR_MAP_FILTER_HPP
#define VECPAR_MAP_FILTER_HPP

#include "vecpar/core/algorithms/detail/filter.hpp"
#include "vecpar/core/algorithms/detail/map.hpp"

namespace vecpar::algorithm {

template <typename R, typename T, typename... Arguments>
struct parallelizable_map_filter
    : public vecpar::detail::parallel_map<R, T, Arguments...>,
      public vecpar::detail::parallel_filter<R> {
  using result_t = R;
  using input_t = T;
  using input_type = vecmem::vector<T>;
  using output_type_t = vecmem::vector<R>;

  TARGET virtual R &map(R &result_i, T &input_i, Arguments... args) = 0;
  TARGET virtual bool filter(R &partial_result) = 0;
};

template <typename TT, typename... Arguments>
struct parallelizable_mmap_filter
    : public vecpar::detail::parallel_mmap<TT, TT, Arguments...>,
      public vecpar::detail::parallel_filter<TT> {
  using result_t = TT;
  using input_t = TT;
  using input_type = vecmem::vector<TT>;
  using output_type_t = vecmem::vector<TT>;

  TARGET virtual TT &map(TT &input_output_i, Arguments... args) = 0;
  TARGET virtual bool filter(TT &partial_result) = 0;
};
} // namespace vecpar::algorithm
#endif // VECPAR_MAP_FILTER_HPP
