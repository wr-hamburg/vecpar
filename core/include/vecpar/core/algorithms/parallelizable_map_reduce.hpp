#ifndef VECPAR_MAP_REDUCE_HPP
#define VECPAR_MAP_REDUCE_HPP

#include "vecpar/core/algorithms/detail/map.hpp"
#include "vecpar/core/algorithms/detail/reduce.hpp"

namespace vecpar::algorithm {

/// R is a collection of <Result>
template <count, typename Result, Iterable R, Iterable T, typename... Arguments>
struct parallelizable_map_reduce {};

template <typename Result, Iterable R, Iterable T, typename... Arguments>
struct parallelizable_map_reduce<One, Result, R, T, Arguments...>
    : public vecpar::detail::parallel_map_one<R, T, Arguments...>,
      public vecpar::detail::parallel_reduce<R> {
  using input_t = T;
  using result_t = Result;
  using intermediate_result_t = R;
};

template <typename Result, Iterable R, Iterable T, typename... Arguments>
struct parallelizable_map_reduce<Two, Result, R, T, Arguments...>
    : public vecpar::detail::parallel_map_two<R, T, Arguments...>,
      public vecpar::detail::parallel_reduce<R> {
  using input_t = T;
  using result_t = Result;
  using intermediate_result_t = R;
};

template <typename Result, Iterable R, Iterable T, typename... Arguments>
struct parallelizable_map_reduce<Three, Result, R, T, Arguments...>
    : public vecpar::detail::parallel_map_three<R, T, Arguments...>,
      public vecpar::detail::parallel_reduce<R> {
  using input_t = T;
  using result_t = Result;
  using intermediate_result_t = R;
};
template <typename Result, Iterable R, Iterable T, typename... Arguments>
struct parallelizable_map_reduce<Four, Result, R, T, Arguments...>
    : public vecpar::detail::parallel_map_four<R, T, Arguments...>,
      public vecpar::detail::parallel_reduce<R> {
  using input_t = T;
  using result_t = Result;
  using intermediate_result_t = R;
};
template <typename Result, Iterable R, Iterable T, typename... Arguments>
struct parallelizable_map_reduce<Five, Result, R, T, Arguments...>
    : public vecpar::detail::parallel_map_five<R, T, Arguments...>,
      public vecpar::detail::parallel_reduce<R> {
  using input_t = T;
  using result_t = Result;
  using intermediate_result_t = R;
};

/// R is a collection of <Result>
template <count, typename Result, Iterable R, typename... Arguments>
struct parallelizable_mmap_reduce {};

template <typename Result, Iterable R, typename... Arguments>
struct parallelizable_mmap_reduce<One, Result, R, Arguments...>
    : public vecpar::detail::parallel_mmap_one<R, Arguments...>,
      public vecpar::detail::parallel_reduce<R> {
  using input_t = R;
  using result_t = Result;
  using intermediate_result_t = R;
};

template <typename Result, Iterable R, typename... Arguments>
struct parallelizable_mmap_reduce<Two, Result, R, Arguments...>
    : public vecpar::detail::parallel_mmap_two<R, Arguments...>,
      public vecpar::detail::parallel_reduce<R> {
  using input_t = R;
  using result_t = Result;
  using intermediate_result_t = R;
};
template <typename Result, Iterable R, typename... Arguments>
struct parallelizable_mmap_reduce<Three, Result, R, Arguments...>
    : public vecpar::detail::parallel_mmap_three<R, Arguments...>,
      public vecpar::detail::parallel_reduce<R> {
  using input_t = R;
  using result_t = Result;
  using intermediate_result_t = R;
};
template <typename Result, Iterable R, typename... Arguments>
struct parallelizable_mmap_reduce<Four, Result, R, Arguments...>
    : public vecpar::detail::parallel_mmap_four<R, Arguments...>,
      public vecpar::detail::parallel_reduce<R> {
  using input_t = R;
  using result_t = Result;
  using intermediate_result_t = R;
};
template <typename Result, Iterable R, typename... Arguments>
struct parallelizable_mmap_reduce<Five, Result, R, Arguments...>
    : public vecpar::detail::parallel_mmap_five<R, Arguments...>,
      public vecpar::detail::parallel_reduce<R> {
  using input_t = R;
  using result_t = Result;
  using intermediate_result_t = R;
};

/// concepts
template <typename Algorithm, typename... All>
concept is_map_reduce_1 =
    std::is_base_of<parallelizable_map_reduce<One, All...>, Algorithm>::value;
template <typename Algorithm, typename... All>

concept is_map_reduce_2 =
    std::is_base_of<parallelizable_map_reduce<Two, All...>, Algorithm>::value;

template <typename Algorithm, typename... All>
concept is_map_reduce_3 =
    std::is_base_of<parallelizable_map_reduce<Three, All...>, Algorithm>::value;

template <typename Algorithm, typename... All>
concept is_map_reduce_4 =
    std::is_base_of<parallelizable_map_reduce<Four, All...>, Algorithm>::value;

template <typename Algorithm, typename... All>
concept is_map_reduce_5 =
    std::is_base_of<parallelizable_map_reduce<Five, All...>, Algorithm>::value;

template <typename Algorithm, typename... All>
concept is_map_reduce = is_map_reduce_1<Algorithm, All...> ||
    is_map_reduce_2<Algorithm, All...> || is_map_reduce_3<Algorithm, All...> ||
    is_map_reduce_4<Algorithm, All...> || is_map_reduce_5<Algorithm, All...>;

template <typename Algorithm, typename... All>
concept is_mmap_reduce_1 =
    std::is_base_of<parallelizable_mmap_reduce<One, All...>, Algorithm>::value;
template <typename Algorithm, typename... All>

concept is_mmap_reduce_2 =
    std::is_base_of<parallelizable_mmap_reduce<Two, All...>, Algorithm>::value;

template <typename Algorithm, typename... All>
concept is_mmap_reduce_3 =
    std::is_base_of<parallelizable_mmap_reduce<Three, All...>,
                    Algorithm>::value;

template <typename Algorithm, typename... All>
concept is_mmap_reduce_4 =
    std::is_base_of<parallelizable_mmap_reduce<Four, All...>, Algorithm>::value;

template <typename Algorithm, typename... All>
concept is_mmap_reduce_5 =
    std::is_base_of<parallelizable_mmap_reduce<Five, All...>, Algorithm>::value;

template <typename Algorithm, typename... All>
concept is_mmap_reduce = is_mmap_reduce_1<Algorithm, All...> ||
    is_mmap_reduce_2<Algorithm, All...> ||
    is_mmap_reduce_3<Algorithm, All...> ||
    is_mmap_reduce_4<Algorithm, All...> || is_mmap_reduce_5<Algorithm, All...>;

} // namespace vecpar::algorithm
#endif // VECPAR_MAP_REDUCE_HPP
