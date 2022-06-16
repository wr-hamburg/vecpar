#ifndef VECPAR_PARALLELIZABLE_FILTER_HPP
#define VECPAR_PARALLELIZABLE_FILTER_HPP

#include "vecpar/core/algorithms/detail/filter.hpp"

namespace vecpar::algorithm {

template <typename R>
struct parallelizable_filter : public vecpar::detail::parallel_filter<R> {
  using result_t = R;
  using input_t = R;

  using output_type_t = vecmem::vector<R>;
  using input_type = vecmem::vector<R>;

  TARGET virtual bool filter(R &partial_result) = 0;
};
} // namespace vecpar::algorithm
#endif // VECPAR_PARALLELIZABLE_FILTER_HPP
