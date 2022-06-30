#ifndef VECPAR_FILTER_HPP
#define VECPAR_FILTER_HPP

#include "vecpar/core/definitions/common.hpp"

namespace vecpar::detail {

template <Iterable T> struct parallel_filter {
  TARGET virtual bool filter(typename T::value_type &item) = 0;
};

/// concepts
template <typename Algorithm, typename T>
concept is_filter =
    std::is_base_of<vecpar::detail::parallel_filter<T>, Algorithm>::value;

} // namespace vecpar::detail

#endif // VECPAR_FILTER_HPP
