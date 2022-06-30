#ifndef VECPAR_PARALLELIZABLE_FILTER_HPP
#define VECPAR_PARALLELIZABLE_FILTER_HPP

#include "vecpar/core/algorithms/detail/filter.hpp"

namespace vecpar::algorithm {

template <Iterable T>
struct parallelizable_filter : public vecpar::detail::parallel_filter<T> {};

/// concepts
template <typename Algorithm, typename T>
concept is_filter = std::is_base_of<parallelizable_filter<T>, Algorithm>::value;

} // namespace vecpar::algorithm
#endif // VECPAR_PARALLELIZABLE_FILTER_HPP
