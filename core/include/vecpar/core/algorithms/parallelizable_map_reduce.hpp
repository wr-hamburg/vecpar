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
    : public vecpar::detail::parallel_map_1<R, T, Arguments...>,
      public vecpar::detail::parallel_reduce<R> {
  using input_t = T;
  using result_t = Result;
  using intermediate_result_t = R;
};

template <typename Result, Iterable R, Iterable T, typename... Arguments>
struct parallelizable_map_reduce<Two, Result, R, T, Arguments...>
    : public vecpar::detail::parallel_map_2<R, T, Arguments...>,
      public vecpar::detail::parallel_reduce<R> {
  using input_t = T;
  using result_t = Result;
  using intermediate_result_t = R;
};

template <typename Result, Iterable R, Iterable T, typename... Arguments>
struct parallelizable_map_reduce<Three, Result, R, T, Arguments...>
    : public vecpar::detail::parallel_map_3<R, T, Arguments...>,
      public vecpar::detail::parallel_reduce<R> {
  using input_t = T;
  using result_t = Result;
  using intermediate_result_t = R;
};
template <typename Result, Iterable R, Iterable T, typename... Arguments>
struct parallelizable_map_reduce<Four, Result, R, T, Arguments...>
    : public vecpar::detail::parallel_map_4<R, T, Arguments...>,
      public vecpar::detail::parallel_reduce<R> {
  using input_t = T;
  using result_t = Result;
  using intermediate_result_t = R;
};
template <typename Result, Iterable R, Iterable T, typename... Arguments>
struct parallelizable_map_reduce<Five, Result, R, T, Arguments...>
    : public vecpar::detail::parallel_map_5<R, T, Arguments...>,
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
    : public vecpar::detail::parallel_mmap_1<R, Arguments...>,
      public vecpar::detail::parallel_reduce<R> {
  using input_t = R;
  using result_t = Result;
  using intermediate_result_t = R;
};

template <typename Result, Iterable R, typename... Arguments>
struct parallelizable_mmap_reduce<Two, Result, R, Arguments...>
    : public vecpar::detail::parallel_mmap_2<R, Arguments...>,
      public vecpar::detail::parallel_reduce<R> {
  using input_t = R;
  using result_t = Result;
  using intermediate_result_t = R;
};
template <typename Result, Iterable R, typename... Arguments>
struct parallelizable_mmap_reduce<Three, Result, R, Arguments...>
    : public vecpar::detail::parallel_mmap_3<R, Arguments...>,
      public vecpar::detail::parallel_reduce<R> {
  using input_t = R;
  using result_t = Result;
  using intermediate_result_t = R;
};
template <typename Result, Iterable R, typename... Arguments>
struct parallelizable_mmap_reduce<Four, Result, R, Arguments...>
    : public vecpar::detail::parallel_mmap_4<R, Arguments...>,
      public vecpar::detail::parallel_reduce<R> {
  using input_t = R;
  using result_t = Result;
  using intermediate_result_t = R;
};
template <typename Result, Iterable R, typename... Arguments>
struct parallelizable_mmap_reduce<Five, Result, R, Arguments...>
    : public vecpar::detail::parallel_mmap_5<R, Arguments...>,
      public vecpar::detail::parallel_reduce<R> {
  using input_t = R;
  using result_t = Result;
  using intermediate_result_t = R;
};

/// concepts
template <typename Algorithm, typename... All>
concept is_map_reduce =
    // map reduce
    std::is_base_of<parallelizable_map_reduce<One, All...>, Algorithm>::value ||
    std::is_base_of<parallelizable_map_reduce<Two, All...>, Algorithm>::value ||
    std::is_base_of<parallelizable_map_reduce<Three, All...>,
                    Algorithm>::value ||
    std::is_base_of<parallelizable_map_reduce<Four, All...>,
                    Algorithm>::value ||
    std::is_base_of<parallelizable_map_reduce<Five, All...>, Algorithm>::value;

template <typename Algorithm, typename... All>
concept is_mmap_reduce =
    // mmap-reduce
    std::is_base_of<parallelizable_mmap_reduce<One, All...>,
                    Algorithm>::value ||
    std::is_base_of<parallelizable_mmap_reduce<Two, All...>,
                    Algorithm>::value ||
    std::is_base_of<parallelizable_mmap_reduce<Three, All...>,
                    Algorithm>::value ||
    std::is_base_of<parallelizable_mmap_reduce<Four, All...>,
                    Algorithm>::value ||
    std::is_base_of<parallelizable_mmap_reduce<Five, All...>, Algorithm>::value;
} // namespace vecpar::algorithm
#endif // VECPAR_MAP_REDUCE_HPP
