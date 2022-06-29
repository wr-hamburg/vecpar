#ifndef VECPAR_PARALLELIZABLE_MAP_HPP
#define VECPAR_PARALLELIZABLE_MAP_HPP

#include "vecpar/core/algorithms/detail/map.hpp"

namespace vecpar::algorithm {
/// 1 iterable collection
template <typename R, typename T, typename... Arguments>
struct parallelizable_map_1
    : public vecpar::detail::parallel_map_1<R, T, Arguments...> {};

/// 1 iterable and mutable collection
template <typename T, typename... Arguments>
struct parallelizable_mmap_1
    : public vecpar::detail::parallel_mmap_1<T, Arguments...> {};

/// 2 iterable collections
template <typename R, typename T1, typename T2, typename... Arguments>
struct parallelizable_map_2
    : public vecpar::detail::parallel_map_2<R, T1, T2, Arguments...> {};

/// 2 iterable collections; result in first collection
template <typename T1, typename T2, typename... Arguments>
struct parallelizable_mmap_2
    : public vecpar::detail::parallel_mmap_2<T1, T2, Arguments...> {};

/// 3 iterable collections
template <typename R, typename T1, typename T2, typename T3,
          typename... Arguments>
struct parallelizable_map_3
    : public vecpar::detail::parallel_map_3<R, T1, T2, T3, Arguments...> {};

/// 3 iterable collections; result in first collection
template <typename T1, typename T2, typename T3, typename... Arguments>
struct parallelizable_mmap_3
    : public vecpar::detail::parallel_mmap_3<T1, T2, T3, Arguments...> {};

/// 4 iterable collections
template <typename R, typename T1, typename T2, typename T3, typename T4,
          typename... Arguments>
struct parallelizable_map_4
    : public vecpar::detail::parallel_map_4<R, T1, T2, T3, T4, Arguments...> {};

/// 4 iterable collections; result in first collection
template <typename T1, typename T2, typename T3, typename T4,
          typename... Arguments>
struct parallelizable_mmap_4
    : public vecpar::detail::parallel_mmap_4<T1, T2, T3, T4, Arguments...> {};

/// 5 iterable collections
template <typename R, typename T1, typename T2, typename T3, typename T4,
          typename T5, typename... Arguments>
struct parallelizable_map_5
    : public vecpar::detail::parallel_map_5<R, T1, T2, T3, T4, T5,
                                            Arguments...> {};

/// 5 iterable collections; result in first collection
template <typename T1, typename T2, typename T3, typename T4, typename T5,
          typename... Arguments>
struct parallelizable_mmap_5
    : public vecpar::detail::parallel_mmap_5<T1, T2, T3, T4, T5, Arguments...> {
};

} // namespace vecpar::algorithm
#endif // VECPAR_PARALLELIZABLE_MAP_HPP
