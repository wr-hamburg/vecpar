#ifndef VECPAR_PARALLELIZABLE_REDUCE_HPP
#define VECPAR_PARALLELIZABLE_REDUCE_HPP

#include "vecpar/core/algorithms/detail/reduce.hpp"

namespace vecpar::algorithm {

template <Iterable R>
struct parallelizable_reduce : public vecpar::detail::parallel_reduce<R> {};

/// concepts
template <typename Algorithm, typename R>
concept is_reduce = std::is_base_of<parallelizable_reduce<R>, Algorithm>::value;

} // namespace vecpar::algorithm
#endif // VECPAR_PARALLELIZABLE_REDUCE_HPP
