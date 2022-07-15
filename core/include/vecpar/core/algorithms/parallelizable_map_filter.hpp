#ifndef VECPAR_MAP_FILTER_HPP
#define VECPAR_MAP_FILTER_HPP

#include "vecpar/core/algorithms/detail/filter.hpp"
#include "vecpar/core/algorithms/detail/map.hpp"

namespace vecpar::algorithm {

template <count, Iterable R, Iterable T, typename... Arguments>
struct parallelizable_map_filter {};

template <Iterable R, Iterable T, typename... Arguments>
struct parallelizable_map_filter<One, R, T, Arguments...>
    : public vecpar::detail::parallel_map_one<R, T, Arguments...>,
      public vecpar::detail::parallel_filter<R> {
  using result_t = R;
};
template <Iterable R, Iterable T, typename... Arguments>
struct parallelizable_map_filter<Two, R, T, Arguments...>
    : public vecpar::detail::parallel_map_two<R, T, Arguments...>,
      public vecpar::detail::parallel_filter<R> {
  using result_t = R;
};
template <Iterable R, Iterable T, typename... Arguments>
struct parallelizable_map_filter<Three, R, T, Arguments...>
    : public vecpar::detail::parallel_map_three<R, T, Arguments...>,
      public vecpar::detail::parallel_filter<R> {
  using result_t = R;
};
template <Iterable R, Iterable T, typename... Arguments>
struct parallelizable_map_filter<Four, R, T, Arguments...>
    : public vecpar::detail::parallel_map_four<R, T, Arguments...>,
      public vecpar::detail::parallel_filter<R> {
  using result_t = R;
};
template <Iterable R, Iterable T, typename... Arguments>
struct parallelizable_map_filter<Five, R, T, Arguments...>
    : public vecpar::detail::parallel_map_five<R, T, Arguments...>,
      public vecpar::detail::parallel_filter<R> {
  using result_t = R;
};

template <count, Iterable R, typename... Arguments>
struct parallelizable_mmap_filter {};

template <Iterable R, typename... Arguments>
struct parallelizable_mmap_filter<One, R, Arguments...>
    : public vecpar::detail::parallel_mmap_one<R, Arguments...>,
      public vecpar::detail::parallel_filter<R> {
  using result_t = R;
};

template <Iterable R, typename... Arguments>
struct parallelizable_mmap_filter<Two, R, Arguments...>
    : public vecpar::detail::parallel_mmap_two<R, Arguments...>,
      public vecpar::detail::parallel_filter<R> {
  using result_t = R;
};
template <Iterable R, typename... Arguments>
struct parallelizable_mmap_filter<Three, R, Arguments...>
    : public vecpar::detail::parallel_mmap_three<R, Arguments...>,
      public vecpar::detail::parallel_filter<R> {
  using result_t = R;
};
template <Iterable R, typename... Arguments>
struct parallelizable_mmap_filter<Four, R, Arguments...>
    : public vecpar::detail::parallel_mmap_four<R, Arguments...>,
      public vecpar::detail::parallel_filter<R> {
  using result_t = R;
};
template <Iterable R, typename... Arguments>
struct parallelizable_mmap_filter<Five, R, Arguments...>
    : public vecpar::detail::parallel_mmap_five<R, Arguments...>,
      public vecpar::detail::parallel_filter<R> {
  using result_t = R;
};

/// concepts

template <typename Algorithm, typename... All>
concept is_map_filter_1 =
    std::is_base_of<parallelizable_map_filter<One, All...>, Algorithm>::value;
template <typename Algorithm, typename... All>

concept is_map_filter_2 =
    std::is_base_of<parallelizable_map_filter<Two, All...>, Algorithm>::value;

template <typename Algorithm, typename... All>
concept is_map_filter_3 =
    std::is_base_of<parallelizable_map_filter<Three, All...>, Algorithm>::value;

template <typename Algorithm, typename... All>
concept is_map_filter_4 =
    std::is_base_of<parallelizable_map_filter<Four, All...>, Algorithm>::value;

template <typename Algorithm, typename... All>
concept is_map_filter_5 =
    std::is_base_of<parallelizable_map_filter<Five, All...>, Algorithm>::value;

template <typename Algorithm, typename... All>
concept is_map_filter = is_map_filter_1<Algorithm, All...> ||
    is_map_filter_2<Algorithm, All...> || is_map_filter_3<Algorithm, All...> ||
    is_map_filter_4<Algorithm, All...> || is_map_filter_5<Algorithm, All...>;

template <typename Algorithm, typename... All>
concept is_mmap_filter_1 =
    std::is_base_of<parallelizable_mmap_filter<One, All...>, Algorithm>::value;
template <typename Algorithm, typename... All>

concept is_mmap_filter_2 =
    std::is_base_of<parallelizable_mmap_filter<Two, All...>, Algorithm>::value;

template <typename Algorithm, typename... All>
concept is_mmap_filter_3 =
    std::is_base_of<parallelizable_mmap_filter<Three, All...>,
                    Algorithm>::value;

template <typename Algorithm, typename... All>
concept is_mmap_filter_4 =
    std::is_base_of<parallelizable_mmap_filter<Four, All...>, Algorithm>::value;

template <typename Algorithm, typename... All>
concept is_mmap_filter_5 =
    std::is_base_of<parallelizable_mmap_filter<Five, All...>, Algorithm>::value;

template <typename Algorithm, typename... All>
concept is_mmap_filter = is_mmap_filter_1<Algorithm, All...> ||
    is_mmap_filter_2<Algorithm, All...> ||
    is_mmap_filter_3<Algorithm, All...> ||
    is_mmap_filter_4<Algorithm, All...> || is_mmap_filter_5<Algorithm, All...>;
} // namespace vecpar::algorithm
#endif // VECPAR_MAP_FILTER_HPP
