#ifndef VECPAR_OMPT_PARALLELIZATION_HPP
#define VECPAR_OMPT_PARALLELIZATION_HPP

#include "omp.h"

#include <vecmem/memory/memory_resource.hpp>

//#pragma omp declare target
#include "vecpar/core/algorithms/parallelizable_filter.hpp"
#include "vecpar/core/algorithms/parallelizable_map.hpp"
#include "vecpar/core/algorithms/parallelizable_map_filter.hpp"
#include "vecpar/core/algorithms/parallelizable_map_reduce.hpp"
#include "vecpar/core/algorithms/parallelizable_reduce.hpp"
//#pragma omp end declare target
#include "vecpar/core/definitions/config.hpp"

#include "vecpar/core/definitions/helper.hpp"

#define BLOCK_SIZE 32

namespace vecpar::ompt {

template <class Algorithm,
          typename R = typename Algorithm::intermediate_result_t, typename T,
          typename... Rest>
requires detail::is_map_1<Algorithm, R, T, Rest...>
R &parallel_map(Algorithm &algorithm,
                __attribute__((unused)) vecmem::memory_resource &mr, T &data,
                Rest &...rest) {

  int size = static_cast<int>(data.size());
  R *vecmem_result = new R(size, &mr);
  value_type_t<R> *map_result = vecmem_result->data();
  value_type_t<T> *d_data = data.data();

#if defined(COMPILE_FOR_DEVICE)
  DEBUG_ACTION(
      printf("[OMPT][map]Attempt to run on device with default config \n");)
  Algorithm *d_alg = (Algorithm *)omp_target_alloc(sizeof(Algorithm), 0);
  omp_target_memcpy(d_alg, &algorithm, sizeof(Algorithm), 0, 0,
                    omp_get_default_device(), omp_get_initial_device());
  // if possible use shared memory
#if _OPENMP >= 202111 and (__clang__ == 1 and __clang_major__ >= 16)
  const int grid_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
#pragma omp target teams is_device_ptr(d_alg) map(to                           \
                                                  : d_data [0:size])           \
    map(from                                                                   \
        : map_result[0:size]) num_teams(grid_size)
  {
    value_type_t<R> buffer[BLOCK_SIZE];
// if the compiler supports OpenMP 5.2 allocators
#pragma omp allocate(buffer) allocator(omp_pteam_mem_alloc)

#pragma omp parallel num_threads(BLOCK_SIZE)
    {
      DEBUG_ACTION(printf("Current: team %d, thread %d; %f \n",
                          omp_get_team_num(), omp_get_thread_num(),
                          buffer[omp_get_thread_num()]);)
      // printf("Running on device? = %d\n", !omp_is_initial_device());

      // all threads use the shared memory for computing the output result
      if (omp_get_team_num() * BLOCK_SIZE + omp_get_thread_num() < size) {
        DEBUG_ACTION(printf("Current: team %d, thread %d; %f \n",
                            omp_get_team_num(), omp_get_thread_num(),
                            buffer[omp_get_thread_num()]);)
        d_alg->mapping_function(
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
#else // no shared memory //
#pragma omp target teams distribute parallel for is_device_ptr(d_alg) \
    map(to:d_data[0:size]) map(from: map_result[0:size])
  for (int i = 0; i < size; i++) {
    //  printf("running on device : %d\n", !omp_is_initial_device());
    d_alg->mapping_function(map_result[i], d_data[i], rest...);
  }
#endif
omp_target_free(d_alg, 0);
#else // defined(COMPILE_FOR_HOST)
  DEBUG_ACTION(printf("[OMPT][map]Running on host with default config \n");)
#pragma omp parallel for
  for (int i = 0; i < size; i++) {
    algorithm.mapping_function(map_result[i], data[i], rest...);
  }
#endif
  return *vecmem_result;
}

    template <class Algorithm,
            typename R = typename Algorithm::intermediate_result_t, typename T1, typename T2,
            typename... Rest>
    requires detail::is_map_2<Algorithm, R, T1, T2, Rest...>
    R &parallel_map(Algorithm &algorithm,
                    __attribute__((unused)) vecmem::memory_resource &mr, T1 &data, T2& in_2,
                    Rest &...rest) {

        int size = static_cast<int>(data.size());
        R *vecmem_result = new R(size, &mr);
        value_type_t<R> *map_result = vecmem_result->data();
        value_type_t<T1> *d_data = data.data();
        value_type_t<T2> *in_2_data = in_2.data();

#if defined(COMPILE_FOR_DEVICE)
        DEBUG_ACTION(
      printf("[OMPT][map]Attempt to run on device with default config \n");)
  Algorithm *d_alg = (Algorithm *)omp_target_alloc(sizeof(Algorithm), 0);
  omp_target_memcpy(d_alg, &algorithm, sizeof(Algorithm), 0, 0,
                    omp_get_default_device(), omp_get_initial_device());
#pragma omp target teams distribute parallel for is_device_ptr(d_alg) \
    map(to:d_data[0:size],in_2_data[0:size]) map(from: map_result[0:size])
  for (int i = 0; i < size; i++) {
    d_alg->mapping_function(map_result[i], d_data[i], in_2_data[i], rest...);
  }
  omp_target_free(d_alg,0);
#else // defined(COMPILE_FOR_HOST)
        DEBUG_ACTION(printf("[OMPT][map]Running on host with default config \n");)
#pragma omp parallel for
        for (int i = 0; i < size; i++) {
            algorithm.mapping_function(map_result[i], data[i], in_2[i], rest...);
        }
#endif
        return *vecmem_result;
    }

    template <class Algorithm,
            typename R = typename Algorithm::intermediate_result_t, typename T1, typename T2, typename T3,
            typename... Rest>
    requires detail::is_map_3<Algorithm, R, T1, T2, T3, Rest...>
    R &parallel_map(Algorithm &algorithm,
                    __attribute__((unused)) vecmem::memory_resource &mr, T1 &data, T2& in_2,T3& in_3,
                    Rest &...rest) {

        int size = static_cast<int>(data.size());
        R *vecmem_result = new R(size, &mr);
        value_type_t<R> *map_result = vecmem_result->data();
        value_type_t<T1> *d_data = data.data();
        value_type_t<T2> *in_2_data = in_2.data();
        value_type_t<T3> *in_3_data = in_3.data();

#if defined(COMPILE_FOR_DEVICE)
        DEBUG_ACTION(
                printf("[OMPT][map]Attempt to run on device with default config \n");)
        Algorithm *d_alg = (Algorithm *)omp_target_alloc(sizeof(Algorithm), 0);
        omp_target_memcpy(d_alg, &algorithm, sizeof(Algorithm), 0, 0,
                          omp_get_default_device(), omp_get_initial_device());
#pragma omp target teams distribute parallel for is_device_ptr(d_alg) \
    map(to:d_data[0:size],in_2_data[0:size], in_3_data[0:size]) map(from: map_result[0:size])
        for (int i = 0; i < size; i++) {
            d_alg->mapping_function(map_result[i], d_data[i], in_2_data[i], in_3_data[i], rest...);
        }
        omp_target_free(d_alg,0);
#else // defined(COMPILE_FOR_HOST)
        DEBUG_ACTION(printf("[OMPT][map]Running on host with default config \n");)
#pragma omp parallel for
        for (int i = 0; i < size; i++) {
            algorithm.mapping_function(map_result[i], data[i], in_2[i], in_3[i], rest...);
        }
#endif
        return *vecmem_result;
    }

// mmap without user config
template <class Algorithm,
          typename R = typename Algorithm::intermediate_result_t, typename T,
          typename... Rest>
requires detail::is_mmap_1<Algorithm, T, Rest...>
R &parallel_map(Algorithm &algorithm,
                __attribute__((unused)) vecmem::memory_resource &mr, T &data,
                Rest &...rest) {

  int size = static_cast<int>(data.size());
  value_type_t<T> *d_data = data.data();

#if defined(COMPILE_FOR_DEVICE)
  // if a GPU is available, use it for the computations
  DEBUG_ACTION(
      printf("[OMPT][mmap]Attempt to run on device with default config \n");)

  Algorithm *d_alg = (Algorithm *)omp_target_alloc(sizeof(Algorithm), 0);
  omp_target_memcpy(d_alg, &algorithm, sizeof(Algorithm), 0, 0,
                          omp_get_default_device(), omp_get_initial_device());
#if _OPENMP >= 202111 and (__clang__ == 1 and __clang_major__ >= 16)
  // use shared memory
  const int grid_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
#pragma omp target teams num_teams(grid_size) is_device_ptr(d_alg)             \
    map(tofrom                                                                 \
        : d_data [0:size])
  // map(to  : rest)
  {
    value_type_t<T> buffer[BLOCK_SIZE];
    // if the compiler supports OpenMP 5.2 allocators
#pragma omp allocate(buffer) allocator(omp_pteam_mem_alloc)

    // thread 0 from each block loads data into shared memory
    for (int i = 0; i < BLOCK_SIZE; i++) {
      if (omp_get_team_num() * BLOCK_SIZE + i < size) {
        buffer[i] = d_data[omp_get_team_num() * BLOCK_SIZE + i];
      }
    }
#pragma omp parallel num_threads(BLOCK_SIZE)
    {
      // all threads use the shared memory for computing the output result
      if (omp_get_team_num() * BLOCK_SIZE + omp_get_thread_num() < size) {
        d_alg->mapping_function(buffer[omp_get_thread_num()], rest...);
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
#else // no shared memory
#pragma omp target teams distribute parallel for is_device_ptr(d_alg)          \
    map(tofrom                                                                 \
        : d_data [0:size]) map(to                                              \
                               : rest) // num_teams(size) num_threads(1)
  for (int i = 0; i < size; i++) {
    d_alg->mapping_function(d_data[i], rest...);
  }
#endif
omp_target_free(d_alg,0);
#else // defined(COMPILE_FOR_HOST)
  DEBUG_ACTION(printf("[OMPT][mmap]Running on host with default config \n");)
#pragma omp parallel for
  for (int i = 0; i < size; i++) {
    algorithm.mapping_function(data[i], rest...);
  }
#endif

  // update the input vector
 // data.assign(d_data, d_data + size);
  return data;
}

    template <class Algorithm,
            typename R = typename Algorithm::intermediate_result_t, typename T1, typename T2,
            typename... Rest>
    requires detail::is_mmap_2<Algorithm, T1, T2, Rest...>
    R &parallel_map(Algorithm &algorithm,
                    __attribute__((unused)) vecmem::memory_resource &mr, T1 &data, T2& in_2,
                    Rest &...rest) {

        int size = static_cast<int>(data.size());
        value_type_t<T1> *d_data = data.data();
        value_type_t<T2> *in_2_data = in_2.data();

#if defined(COMPILE_FOR_DEVICE)
        // if a GPU is available, use it for the computations
        DEBUG_ACTION(
                printf("[OMPT][mmap]Attempt to run on device with default config \n");)

        Algorithm *d_alg = (Algorithm *)omp_target_alloc(sizeof(Algorithm), 0);
        omp_target_memcpy(d_alg, &algorithm, sizeof(Algorithm), 0, 0,
                          omp_get_default_device(), omp_get_initial_device());

#pragma omp target teams distribute parallel for is_device_ptr(d_alg)          \
    map(tofrom                                                                 \
        : d_data[0:size]) map(to:in_2_data[0:size],rest)
        for (int i = 0; i < size; i++) {
            d_alg->mapping_function(d_data[i], in_2_data[i], rest...);
        }
        omp_target_free(d_alg,0);
#else // defined(COMPILE_FOR_HOST)
        DEBUG_ACTION(printf("[OMPT][mmap]Running on host with default config \n");)
#pragma omp parallel for
  for (int i = 0; i < size; i++) {
    algorithm.mapping_function(data[i], in_2[i], rest...);
  }
#endif

        // update the input vector
    //    data.assign(d_data, d_data + size);
        return data;
    }

    template <class Algorithm,
            typename R = typename Algorithm::intermediate_result_t, typename T1, typename T2, typename T3,
            typename... Rest>
    requires detail::is_mmap_3<Algorithm, T1, T2, T3, Rest...>
    R &parallel_map(Algorithm &algorithm,
                    __attribute__((unused)) vecmem::memory_resource &mr, T1 &data,
                    T2& in_2, T3& in_3,
                    Rest &...rest) {

        int size = static_cast<int>(data.size());
        value_type_t<T1> *d_data = data.data();
        value_type_t<T2> *in_2_data = in_2.data();
        value_type_t<T3> *in_3_data = in_3.data();

#if defined(COMPILE_FOR_DEVICE)
        // if a GPU is available, use it for the computations
        DEBUG_ACTION(
                printf("[OMPT][mmap]Attempt to run on device with default config \n");)

        Algorithm *d_alg = (Algorithm *)omp_target_alloc(sizeof(Algorithm), 0);
        omp_target_memcpy(d_alg, &algorithm, sizeof(Algorithm), 0, 0,
                    omp_get_default_device(), omp_get_initial_device());

#pragma omp target teams distribute parallel for is_device_ptr(d_alg)          \
    map(tofrom                                                                 \
        : d_data[0:size]) map(to:in_2_data[0:size], in_3_data[0:size], rest)
        for (int i = 0; i < size; i++) {
            d_alg->mapping_function(d_data[i], in_2_data[i], in_3_data[i], rest...);
        }
        omp_target_free(d_alg,0);
#else // defined(COMPILE_FOR_HOST)
        DEBUG_ACTION(printf("[OMPT][mmap]Running on host with default config \n");)
#pragma omp parallel for
  for (int i = 0; i < size; i++) {
    algorithm.mapping_function(data[i], in_2[i], in_3[i], rest...);
  }
#endif

        // update the input vector
    //    data.assign(d_data, d_data + size);
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

  std::size_t size = data.size();
#if defined(COMPILE_FOR_DEVICE)
  constexpr std::size_t num_target_teams = 40;

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
    if ((std::size_t)(omp_get_team_num() * BLOCK_SIZE + omp_get_thread_num()) <
        size) {
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
  // }
#else // defined(COMPILE_FOR_HOST)
#pragma omp parallel
  {
    data_value_type *temp_result = new data_value_type();

#pragma omp for nowait
    for (std::size_t i = 0; i < size; i++)
      algorithm.reducing_function(temp_result, data[i]);

#pragma omp critical
    algorithm.reducing_function(result, *temp_result);
  }
#endif

  return *result;
}

// filter without user config
template <typename Algorithm, typename T>
requires detail::is_filter<Algorithm, T>
T &parallel_filter(__attribute__((unused)) Algorithm algorithm,
                   vecmem::memory_resource &mr, T &data) {

  T *result;

#if defined(COMPILE_FOR_DEVICE)
  std::size_t num_target_teams = 40;
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
#else // defined(COMPILE_FOR_HOST)
  result = new T(data.size(), &mr);

  std::size_t result_index = 0;
  for (std::size_t i = 0; i < data.size(); i++) {

    if (algorithm.filtering_function(data[i])) {

      result->data()[result_index] = data.data()[i];
      result_index++;
    }
  }

  result->resize(result_index);
#endif

  return *result;
}

//TODO: implement this more efficient using #pragma omp target data
// so that the data is transferred only once and reused by map and reduce
    template <class Algorithm, typename Result, typename R, typename T,
            typename... Arguments>
    Result &parallel_map_reduce(Algorithm &algorithm, vecmem::memory_resource &mr,
                                T &data, Arguments &...args) {
        return vecpar::ompt::parallel_reduce(
                algorithm, mr, vecpar::ompt::parallel_map(algorithm, mr, data, args...));
    }

//TODO: implement this more efficient using #pragma omp target data
// so that the data is transferred only once and reused by map and filter
    template <class Algorithm, typename R, typename T, typename... Arguments>
    R &parallel_map_filter(Algorithm &algorithm, vecmem::memory_resource &mr,
                           T &data, Arguments &...args) {

        return vecpar::ompt::parallel_filter<Algorithm, R>(
                algorithm, mr,
                vecpar::ompt::parallel_map<Algorithm, R, T, Arguments...>(
                        algorithm, mr, data, args...));
    }

    template <class MemoryResource, class Algorithm,
            typename R = typename Algorithm::intermediate_result_t, typename T,
            typename... Arguments>
    requires algorithm::is_map<Algorithm, R, T, Arguments...> ||
             algorithm::is_mmap<Algorithm, R, Arguments...>
    R &parallel_algorithm(Algorithm algorithm, MemoryResource &mr,
                               T &data,
                               Arguments &...args) {

        return vecpar::ompt::parallel_map(algorithm, mr, data, args...);
    }

    template <class MemoryResource, class Algorithm,
            typename Result = typename Algorithm::result_t,
            typename R = typename Algorithm::intermediate_result_t, typename T,
            typename... Arguments>
    requires algorithm::is_map_reduce<Algorithm, Result, R, T, Arguments...> ||
             algorithm::is_mmap_reduce<Algorithm, Result, T, Arguments...>
    Result &parallel_algorithm(Algorithm algorithm, MemoryResource &mr,
                               T &data,
                               Arguments &...args) {

        return vecpar::ompt::parallel_map_reduce<Algorithm, Result, R, T,
                Arguments...>(algorithm, mr, data, args...);
    }

       template <class MemoryResource, class Algorithm,
            class R = typename Algorithm::result_t, typename T,
            typename... Arguments>
    requires algorithm::is_map_filter<Algorithm, R, T, Arguments...> ||
             algorithm::is_mmap_filter<Algorithm, T, Arguments...>
    R &parallel_algorithm(Algorithm algorithm, MemoryResource &mr,
                          T &data,
                          Arguments &...args) {

           return vecpar::ompt::parallel_map_filter<Algorithm, R, T, Arguments...>(
                   algorithm, mr, data, args...);
       }

} // namespace vecpar::ompt
#endif // VECPAR_OMPT_PARALLELIZATION_HPP
