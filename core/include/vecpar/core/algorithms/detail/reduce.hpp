#ifndef VECPAR_REDUCE_HPP
#define VECPAR_REDUCE_HPP

#include "vecpar/core/definitions/common.hpp"

namespace vecpar::detail {

/**
 * The operation has to be commutative and associative
 * since the order is not guaranteed.
 */
template <Iterable R> struct parallel_reduce {
  TARGET typename R::value_type *
  reducing_function(typename R::value_type *result,
                    typename R::value_type &partial_result) const;
};

/// concepts
template <typename Algorithm, typename R>
concept is_reduce =
    std::is_base_of<vecpar::detail::parallel_reduce<R>, Algorithm>::value;

} // namespace vecpar::detail
#endif // VECPAR_REDUCE_HPP
