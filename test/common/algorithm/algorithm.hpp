#ifndef VECPAR_ALGORITHM_HPP
#define VECPAR_ALGORITHM_HPP

#include <functional>
#include <type_traits>
#include <utility>

#include "vecpar/core/definitions/common.hpp"

namespace traccc {
template <typename T> class algorithm {};

template <typename R, typename... A> class algorithm<R(A...)> {
public:
  using output_type = R;

  using function_type = R(A...);

  virtual output_type operator()(A... args) = 0;
};
} // namespace traccc
#endif // VECPAR_ALGORITHM_HPP
