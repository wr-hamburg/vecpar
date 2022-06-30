#ifndef VECPAR_MAP_FILTER_HPP
#define VECPAR_MAP_FILTER_HPP

#include "vecpar/core/algorithms/detail/filter.hpp"
#include "vecpar/core/algorithms/detail/map.hpp"

namespace vecpar::algorithm {

template <count, Iterable R, Iterable T, typename... Arguments>
struct parallelizable_map_filter {};

template <Iterable R, Iterable T, typename... Arguments>
struct parallelizable_map_filter<One, R, T, Arguments...>
    : public vecpar::detail::parallel_map_1<R, T, Arguments...>,
      public vecpar::detail::parallel_filter<R> {
  using result_t = R;
};
template <Iterable R, Iterable T, typename... Arguments>
struct parallelizable_map_filter<Two, R, T, Arguments...>
    : public vecpar::detail::parallel_map_2<R, T, Arguments...>,
      public vecpar::detail::parallel_filter<R> {
  using result_t = R;
};
template <Iterable R, Iterable T, typename... Arguments>
struct parallelizable_map_filter<Three, R, T, Arguments...>
    : public vecpar::detail::parallel_map_3<R, T, Arguments...>,
      public vecpar::detail::parallel_filter<R> {
  using result_t = R;
};
template <Iterable R, Iterable T, typename... Arguments>
struct parallelizable_map_filter<Four, R, T, Arguments...>
    : public vecpar::detail::parallel_map_4<R, T, Arguments...>,
      public vecpar::detail::parallel_filter<R> {
  using result_t = R;
};
template <Iterable R, Iterable T, typename... Arguments>
struct parallelizable_map_filter<Five, R, T, Arguments...>
    : public vecpar::detail::parallel_map_5<R, T, Arguments...>,
      public vecpar::detail::parallel_filter<R> {
  using result_t = R;
};

template <count, Iterable R, typename... Arguments>
struct parallelizable_mmap_filter {};

template <Iterable R, typename... Arguments>
struct parallelizable_mmap_filter<One, R, Arguments...>
    : public vecpar::detail::parallel_mmap_1<R, Arguments...>,
      public vecpar::detail::parallel_filter<R> {
  using result_t = R;
};

template <Iterable R, typename... Arguments>
struct parallelizable_mmap_filter<Two, R, Arguments...>
    : public vecpar::detail::parallel_mmap_1<R, Arguments...>,
      public vecpar::detail::parallel_filter<R> {
  using result_t = R;
};
template <Iterable R, typename... Arguments>
struct parallelizable_mmap_filter<Three, R, Arguments...>
    : public vecpar::detail::parallel_mmap_1<R, Arguments...>,
      public vecpar::detail::parallel_filter<R> {
  using result_t = R;
};
template <Iterable R, typename... Arguments>
struct parallelizable_mmap_filter<Four, R, Arguments...>
    : public vecpar::detail::parallel_mmap_1<R, Arguments...>,
      public vecpar::detail::parallel_filter<R> {
  using result_t = R;
};
template <Iterable R, typename... Arguments>
struct parallelizable_mmap_filter<Five, R, Arguments...>
    : public vecpar::detail::parallel_mmap_1<R, Arguments...>,
      public vecpar::detail::parallel_filter<R> {
  using result_t = R;
};

/// concepts
template <typename Algorithm, typename... All>
concept is_map_filter =
    // map filter
    std::is_base_of<parallelizable_map_filter<One, All...>, Algorithm>::value ||
    std::is_base_of<parallelizable_map_filter<Two, All...>, Algorithm>::value ||
    std::is_base_of<parallelizable_map_filter<Three, All...>,
                    Algorithm>::value ||
    std::is_base_of<parallelizable_map_filter<Four, All...>,
                    Algorithm>::value ||
    std::is_base_of<parallelizable_map_filter<Five, All...>, Algorithm>::value;

template <typename Algorithm, typename... All>
concept is_mmap_filter =
    std::is_base_of<parallelizable_mmap_filter<One, All...>,
                    Algorithm>::value ||
    std::is_base_of<parallelizable_mmap_filter<Two, All...>,
                    Algorithm>::value ||
    std::is_base_of<parallelizable_mmap_filter<Three, All...>,
                    Algorithm>::value ||
    std::is_base_of<parallelizable_mmap_filter<Four, All...>,
                    Algorithm>::value ||
    std::is_base_of<parallelizable_mmap_filter<Five, All...>, Algorithm>::value;

} // namespace vecpar::algorithm
#endif // VECPAR_MAP_FILTER_HPP
