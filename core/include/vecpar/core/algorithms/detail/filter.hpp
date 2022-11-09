#ifndef VECPAR_FILTER_HPP
#define VECPAR_FILTER_HPP

#include "vecpar/core/definitions/common.hpp"
#include "vecpar/core/definitions/types.hpp"

namespace vecpar::detail {

template <vecpar::collection::Iterable R> struct parallel_filter {
  TARGET bool filter(typename R::value_type &item) const;
};

/// concepts
template <typename Algorithm, typename R>
concept is_filter =
    std::is_base_of<vecpar::detail::parallel_filter<R>, Algorithm>::value;

} // namespace vecpar::detail

#endif // VECPAR_FILTER_HPP
