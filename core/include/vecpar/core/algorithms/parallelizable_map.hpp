#ifndef VECPAR_PARALLELIZABLE_MAP_HPP
#define VECPAR_PARALLELIZABLE_MAP_HPP

#include "vecpar/core/algorithms/detail/map.hpp"

namespace vecpar::algorithm {

template <count, typename... Arguments> struct parallelizable_map {};

template <typename... Arguments>
struct parallelizable_map<One, Arguments...>
    : public vecpar::detail::parallel_map_1<Arguments...> {};

template <typename... Arguments>
struct parallelizable_map<Two, Arguments...>
    : public vecpar::detail::parallel_map_2<Arguments...> {};

template <typename... Arguments>
struct parallelizable_map<Three, Arguments...>
    : public vecpar::detail::parallel_map_3<Arguments...> {};

template <typename... Arguments>
struct parallelizable_map<Four, Arguments...>
    : public vecpar::detail::parallel_map_4<Arguments...> {};

template <typename... Arguments>
struct parallelizable_map<Five, Arguments...>
    : public vecpar::detail::parallel_map_5<Arguments...> {};

template <count, typename... Arguments> struct parallelizable_mmap {};

template <typename... Arguments>
struct parallelizable_mmap<One, Arguments...>
    : public vecpar::detail::parallel_mmap_1<Arguments...> {};

template <typename... Arguments>
struct parallelizable_mmap<Two, Arguments...>
    : public vecpar::detail::parallel_mmap_2<Arguments...> {};

template <typename... Arguments>
struct parallelizable_mmap<Three, Arguments...>
    : public vecpar::detail::parallel_mmap_3<Arguments...> {};

template <typename... Arguments>
struct parallelizable_mmap<Four, Arguments...>
    : public vecpar::detail::parallel_mmap_4<Arguments...> {};

template <typename... Arguments>
struct parallelizable_mmap<Five, Arguments...>
    : public vecpar::detail::parallel_mmap_5<Arguments...> {};

/// concepts
template <typename Algorithm, typename... All>
concept is_map =
    std::is_base_of<parallelizable_map<One, All...>, Algorithm>::value ||
    std::is_base_of<parallelizable_map<Two, All...>, Algorithm>::value ||
    std::is_base_of<parallelizable_map<Three, All...>, Algorithm>::value ||
    std::is_base_of<parallelizable_map<Four, All...>, Algorithm>::value ||
    std::is_base_of<parallelizable_map<Five, All...>, Algorithm>::value;

template <typename Algorithm, typename... All>
concept is_mmap =
    std::is_base_of<parallelizable_mmap<One, All...>, Algorithm>::value ||
    std::is_base_of<parallelizable_mmap<Two, All...>, Algorithm>::value ||
    std::is_base_of<parallelizable_mmap<Three, All...>, Algorithm>::value ||
    std::is_base_of<parallelizable_mmap<Four, All...>, Algorithm>::value ||
    std::is_base_of<parallelizable_mmap<Five, All...>, Algorithm>::value;

} // namespace vecpar::algorithm
#endif // VECPAR_PARALLELIZABLE_MAP_HPP
