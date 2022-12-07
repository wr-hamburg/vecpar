#ifndef VECPAR_OMPT_PARALLELIZATION_HPP
#define VECPAR_OMPT_PARALLELIZATION_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdio.h>
#include <type_traits>
#include <utility>

#include "omp.h"

#include <vecmem/memory/memory_resource.hpp>

#pragma omp declare target
#include "vecpar/core/algorithms/parallelizable_filter.hpp"
#include "vecpar/core/algorithms/parallelizable_map.hpp"
#include "vecpar/core/algorithms/parallelizable_map_filter.hpp"
#include "vecpar/core/algorithms/parallelizable_map_reduce.hpp"
#include "vecpar/core/algorithms/parallelizable_reduce.hpp"
#pragma omp end declare target
#include "vecpar/core/definitions/config.hpp"

#include "vecpar/core/definitions/helper.hpp"

/*
#pragma omp metadirective \
      when( device={arch("nvptx")}: teams loop) \
      default( parallel loop)
      */
namespace vecpar::ompt {
template <class Algorithm,
          typename R = typename Algorithm::intermediate_result_t, typename T,
          typename... Rest>
requires detail::is_map<Algorithm, R, T, Rest...>
R &parallel_map(__attribute__((unused)) Algorithm &algorithm,
                __attribute__((unused)) vecmem::memory_resource &mr,
                vecpar::config config, T &data, Rest &...rest) {

  printf("***** OMPT library (variant 1) ***** \n");
  std::size_t size = data.size();

  value_type_t<R> *map_result = new value_type_t<R>[size];
  value_type_t<T> *d_data = data.data();
  Algorithm *d_alg = (Algorithm *)omp_target_alloc(sizeof(Algorithm), 0);

#pragma omp target teams distribute parallel for map(to                        \
                                                     : d_data [0:size])        \
    map(tofrom                                                                 \
        : map_result [0:size]) is_device_ptr(d_alg)                            \
        num_teams(config.m_gridSize) num_threads(config.m_blockSize)

  for (size_t i = 0; i < size; i++) {
    //      printf("%d %f \n", d_data[i], map_result[i]);
    d_alg->map(map_result[i], d_data[i], rest...);
    // printf("Running on device? = %d\n", !omp_is_initial_device());
    //     DEBUG_ACTION(printf("Running on device? = %d\n",
    //     !omp_is_initial_device());)
    // DEBUG_ACTION(printf("Current: team %d, thread %d \n", omp_get_team_num(),
    //                     omp_get_thread_num());)
  }

  omp_target_free(d_alg, 0);

  R *vecmem_result = new R(size, &mr);
  vecmem_result->assign(map_result, map_result + size);

  return *vecmem_result;
}

template <class Algorithm,
          typename R = typename Algorithm::intermediate_result_t, typename T,
          typename... Rest>
requires detail::is_map<Algorithm, R, T, Rest...>
R &parallel_map(__attribute__((unused)) Algorithm &algorithm,
                __attribute__((unused)) vecmem::memory_resource &mr, T &data,
                Rest &...rest) {
  printf("***** OMPT library (variant 2) ***** \n");
  std::size_t size = data.size();

  value_type_t<R> *map_result = new value_type_t<R>[size];
  value_type_t<T> *d_data = data.data();
  Algorithm *d_alg = (Algorithm *)omp_target_alloc(sizeof(Algorithm), 0);

#pragma omp target teams distribute parallel for map(to                        \
                                                     : d_data [0:size])        \
    map(tofrom                                                                 \
        : map_result [0:size]) is_device_ptr(d_alg)
  for (size_t i = 0; i < size; i++) {
    //      printf("%d %f \n", d_data[i], map_result[i]);
    d_alg->map(map_result[i], d_data[i], rest...);
    //   DEBUG_ACTION(printf("Running on device? = %d\n",
    //   !omp_is_initial_device());)
    // printf("Running on device? = %d\n", !omp_is_initial_device());
    // DEBUG_ACTION(printf("Current: team %d, thread %d \n", omp_get_team_num(),
    //                    omp_get_thread_num());)
  }

  omp_target_free(d_alg, 0);
  R *vecmem_result = new R(size, &mr);
  vecmem_result->assign(map_result, map_result + size);
  return *vecmem_result;
}

template <class Algorithm,
          typename R = typename Algorithm::intermediate_result_t, typename T,
          typename... Rest>
requires detail::is_mmap<Algorithm, T, Rest...>
inline R &parallel_map(__attribute__((unused)) Algorithm &algorithm,
                       __attribute__((unused)) vecmem::memory_resource &mr,
                       vecpar::config config, T &data, Rest &...rest) {
  printf("***** OMPT library (variant 3) ***** \n");
  std::size_t size = data.size();

  value_type_t<T> *d_data = data.data();

  Algorithm *d_alg = (Algorithm *)omp_target_alloc(sizeof(Algorithm),
                                                   omp_get_default_device());
#pragma omp target teams distribute parallel for map(tofrom                    \
                                                     : d_data [0:size])        \
    is_device_ptr(d_alg) thread_limit(config.m_blockSize)                      \
        num_teams(config.m_gridSize) num_threads(config.m_blockSize)
  for (size_t i = 0; i < size; i++) {
    // Algorithm d_alg;
    // #pragma omp allocate(d_alg) allocator(omp_cgroup_mem_alloc)
    //   printf("teams = %d, threads per team = %d\n", omp_get_num_teams(),
    //   omp_get_num_threads());
    //      printf("%d %f \n", d_data[i], map_result[i]);
    //      printf("%d ", i);
    d_alg->map(d_data[i], rest...);
    //      printf("Running on device? = %d\n", !omp_is_initial_device());
    //   DEBUG_ACTION(printf("Running on device? = %d\n",
    //   !omp_is_initial_device());)
    // DEBUG_ACTION(printf("Current: team %d, thread %d \n", omp_get_team_num(),
    //                    omp_get_thread_num());)
  }

  omp_target_free(d_alg, omp_get_default_device());
  data.assign(d_data, d_data + size);
  return data;
}

template <class Algorithm,
          typename R = typename Algorithm::intermediate_result_t, typename T,
          typename... Rest>
requires detail::is_mmap<Algorithm, T, Rest...>
R &parallel_map(__attribute__((unused)) Algorithm &algorithm,
                __attribute__((unused)) vecmem::memory_resource &mr, T &data,
                Rest &...rest) {
  printf("***** OMPT library (variant 4) ***** \n");
  std::size_t size = data.size();

  value_type_t<T> *d_data = data.data();
  Algorithm *d_alg = (Algorithm *)omp_target_alloc(sizeof(Algorithm), 0);

#pragma omp target teams distribute parallel for map(tofrom                    \
                                                     : d_data [0:size])        \
    is_device_ptr(d_alg)
  for (size_t i = 0; i < size; i++) {
    //      printf("%d %f \n", d_data[i], map_result[i]);
    d_alg->map(d_data[i], rest...);
    // algorithm.map(d_data[i], rest...);
    // printf("Running on device? = %d\n", !omp_is_initial_device());
    //   DEBUG_ACTION(printf("Running on device? = %d\n",
    //   !omp_is_initial_device());)
    // DEBUG_ACTION(printf("Current: team %d, thread %d \n", omp_get_team_num(),
    //                 omp_get_thread_num());)
  }

  omp_target_free(d_alg, 0);

  data.assign(d_data, d_data + size);
  return data;
}

template <class Algorithm, typename R>
requires detail::is_reduce<Algorithm, R>
typename R::value_type &
parallel_reduce(__attribute__((unused)) Algorithm &algorithm,
                __attribute__((unused)) vecmem::memory_resource &mr, R &data) {

  using data_value_type = typename R::value_type;

  std::size_t num_target_teams = 20;
  std::size_t num_target_threads = 32;

  printf("***** OMPT library (variant 1) ***** \n");
  std::size_t size = data.size();

  value_type_t<R> *d_data = data.data();
  Algorithm *d_alg = (Algorithm *)omp_target_alloc(sizeof(Algorithm), 0);

  data_value_type *temp_result = (data_value_type *)omp_target_alloc(
      sizeof(data_value_type) * num_target_teams * num_target_threads, 0);

  data_value_type team_temp_result[num_target_teams];
  // memset(team_temp_result, 0, sizeof(typename R::value_type) * 2);

#pragma omp target teams map(to                                                \
                             : d_data [0:size])                                \
    map(from                                                                   \
        : team_temp_result [0:num_target_teams])                               \
        is_device_ptr(d_alg, temp_result) num_teams(num_target_teams)
  // num_teams(config.m_gridSize) num_threads(config.m_blockSize)
  {

    // typename R::value_type temp_result[32];

#pragma omp parallel num_threads(num_target_threads)
    if (omp_get_team_num() * num_target_threads + omp_get_thread_num() < size) {
      temp_result[omp_get_team_num() * num_target_threads +
                  omp_get_thread_num()] =
          d_data[omp_get_team_num() * num_target_threads +
                 omp_get_thread_num()];
    }

#pragma omp distribute parallel for num_threads(num_target_threads)
    for (size_t i = num_target_teams * num_target_threads; i < size; i++) {
      //      printf("%d %f \n", d_data[i], map_result[i]);
      // printf("%d %f %f\n", omp_get_thread_num(), temp_result[i], d_data[i]);
      d_alg->reduce(temp_result + (omp_get_team_num() * num_target_threads +
                                   omp_get_thread_num()),
                    d_data[i]);
      // printf("%d %f %f\n", omp_get_thread_num(), temp_result[i], d_data[i]);
      //  printf("Running on device? = %d\n", !omp_is_initial_device());
      //   DEBUG_ACTION(
      //      printf("Running on device? = %d\n", !omp_is_initial_device());)
      //  printf("Index: %ld, Current: team %d, thread %d \n", i,
      //        omp_get_team_num(), omp_get_thread_num());
    }

    size_t j;
    if ((omp_get_team_num() + 1) * num_target_threads < size) {
      j = num_target_threads;
    } else if (omp_get_team_num() * num_target_threads < size) {
      j = size - omp_get_team_num() * num_target_threads;
    } else {
      j = 0;
    }
    // printf("%d, %ld\n", omp_get_team_num(), j);
    while (j > 1) {
      {

#pragma omp parallel num_threads(num_target_threads)
        {
          size_t i = omp_get_thread_num();
          if (i < j / 2) { /*
             printf("Current: team %d, thread %d, value: %f, value2: %f, j: "
                    "%ld,i: %ld, %ld\n",
                    omp_get_team_num(), omp_get_thread_num(), temp_result[i],
                    temp_result[j - (j / 2) + i], j, i, j - (j / 2) + i);*/
            d_alg->reduce(temp_result + i +
                              omp_get_team_num() * num_target_threads,
                          temp_result[omp_get_team_num() * num_target_threads +
                                      j - (j / 2) + i]);
          }
        }
        j = (j + 1) / 2;
      }
    }
    team_temp_result[omp_get_team_num()] =
        temp_result[omp_get_team_num() * num_target_threads];
  }
  omp_target_free(d_alg, 0);
  omp_target_free(temp_result, 0);

  data_value_type *result = new data_value_type();

  for (size_t i = 0;
       i < std::min(num_target_teams,
                    (size + num_target_threads - 1) / num_target_threads);
       i++) {
    // printf("t: %f\n", team_temp_result[i]);
    d_alg->reduce(result, team_temp_result[i]);
    // printf("r: %f\n", *result);
  }

  return *result;
}

template <typename Algorithm, typename T>
requires detail::is_filter<Algorithm, T>
T &parallel_filter(Algorithm algorithm, vecmem::memory_resource &mr, T &data) {

  std::size_t num_target_teams = 5;
  std::size_t num_target_threads = 32;
  std::size_t size = data.size();
  value_type_t<T> *d_data = data.data();
  Algorithm *d_alg = (Algorithm *)omp_target_alloc(sizeof(Algorithm), 0);

  std::size_t data_per_thread = size / (num_target_teams * num_target_threads);
  std::size_t data_per_thread_remainder =
      size % (num_target_teams * num_target_threads);

  std::size_t *thread_offset = (std::size_t *)omp_target_alloc(
      sizeof(std::size_t) * num_target_teams * num_target_threads, 0);
  bool *filter_result = (bool *)omp_target_alloc(sizeof(bool) * size, 0);

  std::size_t *team_offset = new std::size_t[num_target_teams];

#pragma omp target teams map(to                                                \
                             : d_data [0:size])                                \
    map(from                                                                   \
        : team_offset [0:num_target_teams]) num_teams(num_target_teams)        \
        is_device_ptr(d_alg, thread_offset, filter_result)
  {

#pragma omp parallel num_threads(num_target_threads)
    {
      // exececute filter function
      std::size_t global_id =
          omp_get_team_num() * num_target_threads + omp_get_thread_num();

      std::size_t start = data_per_thread * global_id +
                          std::min(data_per_thread_remainder, global_id);
      std::size_t end = data_per_thread * (global_id + 1) +
                        std::min(data_per_thread_remainder, global_id + 1);

      // printf("1: %d, %d, %ld, %ld, %ld\n", omp_get_team_num(),
      //        omp_get_thread_num(), global_id, start, end);

      std::size_t count = 0;
      for (std::size_t i = start; i < end; i++) {

        bool temp = d_alg->filter(d_data[i]);
        filter_result[i] = temp;

        if (temp) {
          count++;
        }
      }

      // printf("2: %d, %d, %ld, %ld\n", omp_get_team_num(),
      // omp_get_thread_num(),
      //        global_id, count);
      thread_offset[global_id] = count;
    }

    // claculate size and offset in each team

    std::size_t count = 0;

    for (std::size_t i = 0; i < num_target_threads; i++) {

      std::size_t t =
          thread_offset[omp_get_team_num() * num_target_threads + i];

      // printf("3: %d, %d, %ld, %ld\n", omp_get_team_num(),
      //        omp_get_team_num() * num_target_threads + i, i, count);
      thread_offset[omp_get_team_num() * num_target_threads + i] = count;
      count += t;
    }

    team_offset[omp_get_team_num()] = count;
  }

  // calculate size and offset on host
  std::size_t count = 0;
  for (std::size_t i = 0; i < num_target_teams; i++) {
    std::size_t t = team_offset[i];
    team_offset[i] = count;
    count += t;
  }

  // printf("%ld\n", count);

  // allocate buffer for result
  T *result = new T(count, &mr);

  value_type_t<T> *d_result = result->data();
  // move stuff to buffer

#pragma omp target teams map(to                                                \
                             : team_offset [0:num_target_teams])               \
    map(to                                                                     \
        : d_data [0:size]) map(from                                            \
                               : d_result [0:count])                           \
        num_teams(num_target_teams)                                            \
            is_device_ptr(thread_offset, filter_result)
  {

#pragma omp parallel num_threads(num_target_threads)
    {
      std::size_t global_id =
          omp_get_team_num() * num_target_threads + omp_get_thread_num();

      std::size_t start = data_per_thread * global_id +
                          std::min(data_per_thread_remainder, global_id);
      std::size_t end = data_per_thread * (global_id + 1) +
                        std::min(data_per_thread_remainder, global_id + 1);

      std::size_t offset =
          team_offset[omp_get_team_num()] + thread_offset[global_id];

      // printf("4: %d, %d, %ld, %ld, %ld, %ld\n", omp_get_team_num(),
      //        omp_get_thread_num(), global_id, offset,
      //      team_offset[omp_get_team_num()], thread_offset[global_id]);
      for (std::size_t i = start; i < end; i++) {

        if (filter_result[i]) {

          // printf("5: %d, %d, %ld, %ld, %ld\n", omp_get_team_num(),
          //       omp_get_thread_num(), global_id, i, offset);
          d_result[offset] = d_data[i];

          offset++;
        }
      }
    }
  }
  omp_target_free(d_alg, 0);
  omp_target_free(thread_offset, 0);
  omp_target_free(filter_result, 0);
  return *result;
}
} // namespace vecpar::ompt
#endif // VECPAR_OMPT_PARALLELIZATION_HPP
