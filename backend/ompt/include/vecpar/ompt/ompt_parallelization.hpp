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

#define BLOCK_SIZE 32

namespace vecpar::ompt {

// map with user config
/*
template <class Algorithm,
          typename R = typename Algorithm::intermediate_result_t, typename T,
          typename... Rest>
requires detail::is_map<Algorithm, R, T, Rest...>
R &parallel_map(Algorithm &algorithm, vecmem::memory_resource &mr,
                vecpar::config config, T &data, Rest &...rest) {

  printf("Map \n");
  int size = static_cast<int>(data.size());
  value_type_t<R> *map_result = new value_type_t<R>[size];
  value_type_t<T> *d_data = data.data();

  int device = omp_get_num_devices();
  if (device >= 1) {
    DEBUG_ACTION(
        printf("[OMPT][map]Attempt to run on device with user config \n");)
#pragma omp target map(alloc                                                   \
                       : map_result [0:size]) map(from                         \
                                                  : map_result [0:size])       \
    map(to                                                                     \
        : d_data [0:size], rest)
#pragma omp teams distribute parallel for num_teams(config.m_gridSize)         \
    num_threads(config.m_blockSize)
    for (int i = 0; i < size; i++) {
      Algorithm d_alg;
      d_alg.mapping_function(map_result[i], d_data[i], rest...);
      //      printf("Running on device? = %d\n", !omp_is_initial_device());
      // DEBUG_ACTION(printf("Running on device? = %d\n",
      // !omp_is_initial_device());)
      DEBUG_ACTION(printf("Current: team %d, thread %d \n", omp_get_team_num(),
                          omp_get_thread_num());)
    }
  } else { // no GPU available, compute the results on the CPU
    DEBUG_ACTION(printf("[OMPT][map]Running on host with user config \n");)
#pragma omp parallel for num_threads(config.m_blockSize)
    for (int i = 0; i < size; i++) {
      algorithm.mapping_function(map_result[i], data[i], rest...);
    }
  }

  R *vecmem_result = new R(size, &mr);
  vecmem_result->assign(map_result, map_result + size);
  return *vecmem_result;
}
*/

// map without user config
template <class Algorithm,
          typename R = typename Algorithm::intermediate_result_t, typename T,
          typename... Rest>
requires detail::is_map<Algorithm, R, T, Rest...>
R &parallel_map(__attribute__((unused)) Algorithm &algorithm,
                __attribute__((unused)) vecmem::memory_resource &mr, T &data,
                Rest &...rest) {

  // printf("test 2\n");
  int size = static_cast<int>(data.size());
  value_type_t<R> *map_result = new value_type_t<R>[size];
  value_type_t<T> *d_data = data.data();

  // check if GPU is available at runtime
  int device = omp_get_num_devices();

  if (device >= 1) { // if a GPU is available, use it for the computations
    DEBUG_ACTION(
        printf("[OMPT][map]Attempt to run on device with default config \n");)
    const int grid_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // printf("%d\n", grid_size);

#pragma omp target teams map(to                                                \
                             : d_data [0:size], rest)                          \
    map(from                                                                   \
        : map_result [0:size]) num_teams(grid_size)
    {
      value_type_t<R> buffer[BLOCK_SIZE];
      // #pragma omp allocate(buffer) allocator(omp_pteam_mem_alloc)

#pragma omp parallel num_threads(BLOCK_SIZE)
      {

        Algorithm d_alg;
        // #pragma omp allocate(d_alg) allocator(omp_thread_mem_alloc)

        DEBUG_ACTION(printf("Current: team %d, thread %d; %f \n",
                            omp_get_team_num(), omp_get_thread_num(),
                            buffer[omp_get_thread_num()]);)
        // printf("Running on device? = %d\n", !omp_is_initial_device());

        // all threads use the shared memory for computing the output result
        if (omp_get_team_num() * BLOCK_SIZE + omp_get_thread_num() < size) {
          DEBUG_ACTION(printf("Current: team %d, thread %d; %f \n",
                              omp_get_team_num(), omp_get_thread_num(),
                              buffer[omp_get_thread_num()]);)
          d_alg.mapping_function(
              buffer[omp_get_thread_num()],
              d_data[omp_get_team_num() * BLOCK_SIZE + omp_get_thread_num()],
              rest...);
        }
      }

      // thread 0 from each block copies the results from shared memory to
      // global memory
      for (int i = 0; i < BLOCK_SIZE; i++) {
        if (omp_get_team_num() * BLOCK_SIZE + i < size) {
          map_result[omp_get_team_num() * BLOCK_SIZE + i] = buffer[i];
        }
      }
    }
  } else { // no GPU available, compute the results on the CPU
    DEBUG_ACTION(printf("[OMPT][map]Running on host with default config \n");)
#pragma omp parallel for
    for (int i = 0; i < size; i++) {
      algorithm.mapping_function(map_result[i], data[i], rest...);
    }
  }

  R *vecmem_result = new R(size, &mr);
  vecmem_result->assign(map_result, map_result + size);
  return *vecmem_result;
}

// mmap with user config
/*
template <class Algorithm,
          typename R = typename Algorithm::intermediate_result_t, typename T,
          typename... Rest>
requires detail::is_mmap<Algorithm, T, Rest...>
inline R &parallel_map(Algorithm &algorithm,
                       __attribute__((unused)) vecmem::memory_resource &mr,
                       vecpar::config config, T &data, Rest &...rest) {
  printf("test 4\n");
  //    printf("***** OMPT library ***** \n");
  int size = static_cast<int>(data.size());
  value_type_t<T> *d_data = data.data();

  int device = omp_get_num_devices();
  if (device >= 1) { // if a GPU is available, use it for the computations
    DEBUG_ACTION(
        printf("[OMPT][mmap]Attempt to run on device with user config \n");)
#pragma omp target map(tofrom : d_data [0:size]) map(to : rest)
#pragma omp teams distribute parallel for num_teams(config.m_gridSize)         \
    num_threads(config.m_blockSize)
    for (int i = 0; i < size; i++) {
      Algorithm d_alg;
      d_alg.mapping_function(d_data[i], rest...);
      //      printf("Running on device? = %d\n", !omp_is_initial_device());
      //   DEBUG_ACTION(printf("Running on device? = %d\n",
      //   !omp_is_initial_device());)
      DEBUG_ACTION(printf("Current: team %d, thread %d \n", omp_get_team_num(),
                          omp_get_thread_num());)
    }
  } else { // no GPU available, compute the results on the CPU
    DEBUG_ACTION(printf("[OMPT][mmap]Running on host with user config \n");)
#pragma omp parallel for num_threads(config.m_blockSize)
    for (int i = 0; i < size; i++) {
      algorithm.mapping_function(data[i], rest...);
    }
  }

  data.assign(d_data, d_data + size);
  return data;
}
*/

// mmap without user config
template <class Algorithm,
          typename R = typename Algorithm::intermediate_result_t, typename T,
          typename... Rest>
requires detail::is_mmap<Algorithm, T, Rest...>
R &parallel_map(Algorithm &algorithm,
                __attribute__((unused)) vecmem::memory_resource &mr, T &data,
                Rest &...rest) {

  // printf("test 5\n");
  int size = static_cast<int>(data.size());
  value_type_t<T> *d_data = data.data();

  // check if GPU is available at runtime
  int device = omp_get_num_devices();
  if (device >= 1) {
    // if a GPU is available, use it for the computations
    DEBUG_ACTION(
        printf("[OMPT][mmap]Attempt to run on device with default config \n");)
    const int grid_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

#pragma omp target teams num_teams(grid_size) map(tofrom                       \
                                                  : d_data [0:size])           \
    map(to                                                                     \
        : rest)
    {

      value_type_t<T> buffer[BLOCK_SIZE];
      // #pragma omp allocate(buffer) allocator(omp_pteam_mem_alloc)
      // thread 0 from each block loads data into shared memory
      for (int i = 0; i < BLOCK_SIZE; i++) {
        if (omp_get_team_num() * BLOCK_SIZE + i < size) {
          buffer[i] = d_data[omp_get_team_num() * BLOCK_SIZE + i];
        }
      }
#pragma omp parallel num_threads(BLOCK_SIZE)
      {

        Algorithm d_alg;
        //  #pragma omp allocate(d_alg) allocator(omp_thread_mem_alloc)

        // all threads use the shared memory for computing the output result
        if (omp_get_team_num() * BLOCK_SIZE + omp_get_thread_num() < size) {
          d_alg.mapping_function(buffer[omp_get_thread_num()], rest...);
        }
      }
      // thread 0 from each block copies the results from shared memory to
      // global memory
      for (int i = 0; i < BLOCK_SIZE; i++) {
        if (omp_get_team_num() * BLOCK_SIZE + i < size) {
          d_data[omp_get_team_num() * BLOCK_SIZE + i] = buffer[i];
        }
      }
    }
  } else { // no GPU available, compute the results on the CPU
    DEBUG_ACTION(printf("[OMPT][mmap]Running on host with deault config \n");)
#pragma omp parallel for
    for (int i = 0; i < size; i++) {
      algorithm.mapping_function(data[i], rest...);
    }
  }

  // update the input vectorauf
  data.assign(d_data, d_data + size);
  return data;
}

// reduce without user config
template <class Algorithm, typename R>
requires detail::is_reduce<Algorithm, R>
typename R::value_type &
parallel_reduce(__attribute__((unused)) Algorithm &algorithm,
                __attribute__((unused)) vecmem::memory_resource &mr, R &data) {

  using data_value_type = typename R::value_type;
  data_value_type *result = new data_value_type();

  //  constexpr std::size_t num_target_teams = 20;

  std::size_t size = data.size();

  int device = omp_get_num_devices();
  if (device == 0) {

#pragma omp parallel
    {
      data_value_type *temp_result = new data_value_type();

#pragma omp for nowait
      for (std::size_t i = 0; i < size; i++)
        algorithm.reducing_function(temp_result, data[i]);

#pragma omp critical
      algorithm.reducing_function(result, *temp_result);
    }

  } else {

    constexpr std::size_t num_target_teams = 1;

    // printf("***** OMPT library (variant 1) ***** \n");

    value_type_t<R> *d_data = data.data();

    // data_value_type *temp_result = (data_value_type *)omp_target_alloc(
    //     sizeof(data_value_type) * num_target_teams * BLOCK_SIZE, 0);

    data_value_type team_temp_result[num_target_teams];
    // memset(team_temp_result, 0, sizeof(typename R::value_type) * 2);

#pragma omp target teams map(to                                                \
                             : d_data [0:size])                                \
    map(from                                                                   \
        : team_temp_result [0:num_target_teams]) num_teams(num_target_teams)
    // is_device_ptr(temp_result)
    // num_teams(config.m_gridSize) num_threads(config.m_blockSize)
    {

      typename R::value_type temp_result[BLOCK_SIZE];

#pragma omp parallel num_threads(BLOCK_SIZE)
      if ((std::size_t)(omp_get_team_num() * BLOCK_SIZE +
                        omp_get_thread_num()) < size) {
        temp_result[omp_get_thread_num()] =
            d_data[omp_get_team_num() * BLOCK_SIZE + omp_get_thread_num()];
      }

#pragma omp distribute parallel for num_threads(BLOCK_SIZE)
      for (size_t i = num_target_teams * BLOCK_SIZE; i < size; i++) {
        //      printf("%d %f \n", d_data[i], map_result[i]);
        // printf("%d %f %f\n", omp_get_thread_num(), temp_result[i],
        // d_data[i]);
        algorithm.reducing_function(temp_result + omp_get_thread_num(),
                                    d_data[i]);
        // printf("%d %f %f\n", omp_get_thread_num(), temp_result[i],
        // d_data[i]);
        //  printf("Running on device? = %d\n", !omp_is_initial_device());
        //   DEBUG_ACTION(
        //      printf("Running on device? = %d\n", !omp_is_initial_device());)
        //  printf("Index: %ld, Current: team %d, thread %d \n", i,
        //        omp_get_team_num(), omp_get_thread_num());
      }

      size_t j;
      if ((std::size_t)((omp_get_team_num() + 1) * BLOCK_SIZE) < size) {
        j = BLOCK_SIZE;
      } else if ((std::size_t)(omp_get_team_num() * BLOCK_SIZE) < size) {
        j = size - omp_get_team_num() * BLOCK_SIZE;
      } else {
        j = 0;
      }
      // printf("%d, %ld\n", omp_get_team_num(), j);
      while (j > 1) {
        {
#pragma omp parallel num_threads(BLOCK_SIZE)
          {
            size_t i = omp_get_thread_num();
            if (i < j / 2) {
              /*printf("Current: team %d, thread %d, value: %f, value2: %f, j: "
                     "%ld,i: %ld, %ld\n",
                     omp_get_team_num(), omp_get_thread_num(), temp_result[i],
                     temp_result[j - (j / 2) + i], j, i, j - (j / 2) + i);*/
              algorithm.reducing_function(temp_result + i,
                                          temp_result[j - (j / 2) + i]);
            }
          }
          j = (j + 1) / 2;
        }
      }
      team_temp_result[omp_get_team_num()] = temp_result[0];
    }

    for (size_t i = 0;
         i < std::min(num_target_teams, (size + BLOCK_SIZE - 1) / BLOCK_SIZE);
         i++) {
      // printf("t: %f\n", team_temp_result[i]);
      algorithm.reducing_function(result, team_temp_result[i]);
      // printf("r: %f\n", *result);
    }
  }
  return *result;
}

// filter without user config
template <typename Algorithm, typename T>
requires detail::is_filter<Algorithm, T>
T &parallel_filter(__attribute__((unused)) Algorithm algorithm,
                   vecmem::memory_resource &mr, T &data) {

  std::size_t num_target_teams = 5;
  T *result;
  int device = omp_get_num_devices();
  if (device == 0) {

    result = new T(data.size(), &mr);

    std::size_t result_index = 0;
    for (std::size_t i = 0; i < data.size(); i++) {

      if (algorithm.filtering_function(data[i])) {

        result->data()[result_index] = data.data()[i];
        result_index++;
      }
    }

    result->resize(result_index);

  } else {

    std::size_t size = data.size();
    value_type_t<T> *d_data = data.data();

    std::size_t data_per_thread = size / (num_target_teams * BLOCK_SIZE);
    std::size_t data_per_thread_remainder =
        size % (num_target_teams * BLOCK_SIZE);

    std::size_t *thread_offset = new std::size_t[num_target_teams * BLOCK_SIZE];
    bool *filter_result = NULL;

    std::size_t *team_offset = new std::size_t[num_target_teams];

#pragma omp target data map(alloc                                              \
                            : filter_result [0:size],                          \
                              thread_offset [0:num_target_teams * BLOCK_SIZE])
    {

#pragma omp target teams map(to                                                \
                             : d_data [0:size])                                \
    map(from                                                                   \
        : team_offset [0:num_target_teams]) num_teams(num_target_teams)
      {

#pragma omp parallel num_threads(BLOCK_SIZE)
        {
          // exececute filter function
          std::size_t global_id =
              omp_get_team_num() * BLOCK_SIZE + omp_get_thread_num();

          std::size_t start = data_per_thread * global_id +
                              std::min(data_per_thread_remainder, global_id);
          std::size_t end = data_per_thread * (global_id + 1) +
                            std::min(data_per_thread_remainder, global_id + 1);

          // printf("1: %d, %d, %ld, %ld, %ld\n", omp_get_team_num(),
          //        omp_get_thread_num(), global_id, start, end);

          std::size_t count = 0;
          for (std::size_t i = start; i < end; i++) {

            bool temp = algorithm.filtering_function(d_data[i]);
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

        for (std::size_t i = 0; i < BLOCK_SIZE; i++) {

          std::size_t t = thread_offset[omp_get_team_num() * BLOCK_SIZE + i];

          // printf("3: %d, %d, %ld, %ld\n", omp_get_team_num(),
          //        omp_get_team_num() * BLOCK_SIZE + i, i, count);
          thread_offset[omp_get_team_num() * BLOCK_SIZE + i] = count;
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

      // allocate buffer for result
      result = new T(count, &mr);

      value_type_t<T> *d_result = result->data();
      // move stuff to buffer

#pragma omp target teams map(to                                                \
                             : team_offset [0:num_target_teams])               \
    map(to                                                                     \
        : d_data [0:size]) map(from                                            \
                               : d_result [0:count])                           \
        num_teams(num_target_teams)
      {

#pragma omp parallel num_threads(BLOCK_SIZE)
        {
          std::size_t global_id =
              omp_get_team_num() * BLOCK_SIZE + omp_get_thread_num();

          std::size_t start = data_per_thread * global_id +
                              std::min(data_per_thread_remainder, global_id);
          std::size_t end = data_per_thread * (global_id + 1) +
                            std::min(data_per_thread_remainder, global_id + 1);

          std::size_t offset =
              team_offset[omp_get_team_num()] + thread_offset[global_id];

          /*
          printf("4: %d, %d, %ld, %ld, %ld, %ld\n", omp_get_team_num(),
                 omp_get_thread_num(), global_id, offset,
                 team_offset[omp_get_team_num()], thread_offset[global_id]);*/
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
    }
  }
  return *result;
}
} // namespace vecpar::ompt
#endif // VECPAR_OMPT_PARALLELIZATION_HPP
