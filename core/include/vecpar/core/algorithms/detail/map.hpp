#ifndef VECPAR_MAP_HPP
#define VECPAR_MAP_HPP

#include "vecpar/core/definitions/common.hpp"

namespace vecpar::detail {

/// 1 iterable collection
template <typename R, typename T, typename... Arguments> struct parallel_map_1 {
  TARGET virtual typename R::value_type &
  map(typename R::value_type &out_item, const typename T::value_type &in_item,
      Arguments... obj) = 0;
  using input_t = T;
  using input_ti = typename T::value_type;
  using result_t = R;
  using result_ti = typename R::value_type;
  using intermediate_result_t = R;
};

/// 1 iterable and mutable collection
template <typename T, typename... Arguments> struct parallel_mmap_1 {
  TARGET virtual typename T::value_type &
  map(typename T::value_type &in_out_item, Arguments... obj) = 0;
  using input_t = T;
  using input_ti = typename T::value_type;
  using result_t = T;
  using result_ti = typename T::value_type;
  using intermediate_result_t = T;
};

/// 2 iterable collections
template <typename R, typename T1, typename T2, typename... Arguments>
struct parallel_map_2 {
  TARGET virtual typename R::value_type &
  map(typename R::value_type &out_item,
      const typename T1::value_type &in_1_item,
      const typename T2::value_type &in_2_item, Arguments... obj) = 0;
  using result_t = R;
  using intermediate_result_t = R;
};

/// 2 iterable collections; result in first collection
template <typename T1, typename T2, typename... Arguments>
struct parallel_mmap_2 {
  TARGET virtual typename T1::value_type &
  map(typename T1::value_type &in_out_item,
      const typename T2::value_type &in_2_item, Arguments... obj) = 0;
  using result_t = T1;
  using intermediate_result_t = T1;
};

/// 3 iterable collections
template <typename R, typename T1, typename T2, typename T3,
          typename... Arguments>
struct parallel_map_3 {
  TARGET virtual typename R::value_type &
  map(typename R::value_type &out_item,
      const typename T1::value_type &in_1_item,
      const typename T2::value_type &in_2_item,
      const typename T3::value_type &in_3_item, Arguments... obj) = 0;
  using result_t = R;
  using intermediate_result_t = R;
};

/// 3 iterable collections; result in first collection
template <typename T1, typename T2, typename T3, typename... Arguments>
struct parallel_mmap_3 {
  TARGET virtual typename T1::value_type &
  map(typename T1::value_type &in_out_item,
      const typename T2::value_type &in_2_item,
      const typename T3::value_type &in_3_item, Arguments... obj) = 0;
  using result_t = T1;
  using intermediate_result_t = T1;
};

/// 4 iterable collections
template <typename R, typename T1, typename T2, typename T3, typename T4,
          typename... Arguments>
struct parallel_map_4 {
  TARGET virtual typename R::value_type &
  map(typename R::value_type &out_item,
      const typename T1::value_type &in_1_item,
      const typename T2::value_type &in_2_item,
      const typename T3::value_type &in_3_item,
      const typename T4::value_type &in_4_item, Arguments... obj) = 0;
  using result_t = R;
  using intermediate_result_t = R;
};

/// 4 iterable collections; result in first collection
template <typename T1, typename T2, typename T3, typename T4,
          typename... Arguments>
struct parallel_mmap_4 {
  TARGET virtual typename T1::value_type &
  map(typename T1::value_type &in_out_item,
      const typename T2::value_type &in_2_item,
      const typename T3::value_type &in_3_item,
      const typename T4::value_type &in_4_item, Arguments... obj) = 0;
  using result_t = T1;
  using intermediate_result_t = T1;
};

/// 5 iterable collections
template <typename R, typename T1, typename T2, typename T3, typename T4,
          typename T5, typename... Arguments>
struct parallel_map_5 {
  TARGET virtual typename R::value_type &
  map(typename R::value_type &out_item,
      const typename T1::value_type &in_1_item,
      const typename T2::value_type &in_2_item,
      const typename T3::value_type &in_3_item,
      const typename T4::value_type &in_4_item,
      const typename T5::value_type &in_5_item, Arguments... obj) = 0;
  using result_t = R;
  using intermediate_result_t = R;
};

/// 5 iterable collections; result in first collection
template <typename T1, typename T2, typename T3, typename T4, typename T5,
          typename... Arguments>
struct parallel_mmap_5 {
  TARGET virtual typename T1::value_type &
  map(typename T1::value_type &in_out_item,
      const typename T2::value_type &in_2_item,
      const typename T3::value_type &in_3_item,
      const typename T4::value_type &in_4_item,
      const typename T5::value_type &in_5_item, Arguments... obj) = 0;
  using result_t = T1;
  using intermediate_result_t = T1;
};

} // namespace vecpar::detail
#endif // VECPAR_MAP_HPP
