#ifndef VECPAR_PARALLELIZABLE_FILTER_HPP
#define VECPAR_PARALLELIZABLE_FILTER_HPP

#include "vecpar/core/algorithms/detail/filter.hpp"

namespace vecpar::algorithm {

template <typename T>
struct parallelizable_filter : public vecpar::detail::parallel_filter<T> {};

} // namespace vecpar::algorithm
#endif // VECPAR_PARALLELIZABLE_FILTER_HPP
