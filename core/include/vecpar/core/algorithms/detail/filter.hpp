#ifndef VECPAR_FILTER_HPP
#define VECPAR_FILTER_HPP

#include "vecpar/core/definitions/common.hpp"

namespace vecpar::detail {

template <typename T> struct parallel_filter {
  TARGET virtual bool filter(typename T::value_type &item) = 0;
};
} // namespace vecpar::detail

#endif // VECPAR_FILTER_HPP
