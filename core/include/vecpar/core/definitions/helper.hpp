#ifndef VECPAR_HELPER_HPP
#define VECPAR_HELPER_HPP

#include "vecpar/core/definitions/types.hpp"

template <Iterable i>
static inline auto get(int idx, i &collection) -> typename i::value_type & {
  return collection[idx];
}

template <typename Object>
static inline auto get(__attribute__((unused)) int idx, Object &o) -> Object & {
  return o;
}

#endif // VECPAR_HELPER_HPP
