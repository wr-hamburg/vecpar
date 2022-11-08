#ifndef VECPAR_OMPT_PARALLELIZATION_HPP
#define VECPAR_OMPT_PARALLELIZATION_HPP

#include <cmath>
#include <type_traits>
#include <utility>
#include <stdio.h>

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
    template<class Algorithm,
            typename R = typename Algorithm::intermediate_result_t, typename T,
            typename... Rest>
    requires detail::is_map<Algorithm, R, T, Rest...> R &
    parallel_map(__attribute__((unused)) Algorithm &algorithm,
                 __attribute__((unused)) vecmem::memory_resource &mr,
                 vecpar::config config,
                 T &data,
                 Rest&... rest) {

        printf("***** OMPT library ***** \n");
        std::size_t size = data.size();

        value_type_t<R> *map_result = new value_type_t<R>[size];
        value_type_t<T> *d_data = data.data();
        Algorithm *d_alg = (Algorithm *) omp_target_alloc(sizeof(Algorithm), 0);

#pragma omp target teams distribute parallel for \
        map(to:d_data[0:size]) map(tofrom:map_result[0:size]) is_device_ptr(d_alg)  \
    num_teams(config.m_gridSize) num_threads(config.m_blockSize)

        for (size_t i = 0; i < size; i++) {
            //      printf("%d %f \n", d_data[i], map_result[i]);
            d_alg->map(map_result[i], d_data[i], rest...);
            printf("Running on device? = %d\n", !omp_is_initial_device());
        //    DEBUG_ACTION(printf("Running on device? = %d\n", !omp_is_initial_device());)
            DEBUG_ACTION(printf("Current: team %d, thread %d \n", omp_get_team_num(), omp_get_thread_num());)
        }

        omp_target_free(d_alg, 0);

        R *vecmem_result = new R(size, &mr);
        vecmem_result->assign(map_result, map_result + size);

        return *vecmem_result;
    }

    template<class Algorithm,
            typename R = typename Algorithm::intermediate_result_t, typename T,
            typename... Rest>
    requires detail::is_map<Algorithm, R, T, Rest...> R &
    parallel_map(__attribute__((unused)) Algorithm &algorithm,
                 __attribute__((unused)) vecmem::memory_resource &mr,
                 T &data,
                 Rest&... rest) {
        printf("***** OMPT library ***** \n");
        std::size_t size = data.size();

        value_type_t<R> *map_result = new value_type_t<R>[size];
        value_type_t<T> *d_data = data.data();
        Algorithm *d_alg = (Algorithm *) omp_target_alloc(sizeof(Algorithm), 0);

#pragma omp target teams distribute parallel for \
    map(to:d_data[0:size]) map(tofrom:map_result[0:size]) is_device_ptr(d_alg)
        for (size_t i = 0; i < size; i++) {
            //      printf("%d %f \n", d_data[i], map_result[i]);
            d_alg->map(map_result[i], d_data[i], rest...);
         //   DEBUG_ACTION(printf("Running on device? = %d\n", !omp_is_initial_device());)
            printf("Running on device? = %d\n", !omp_is_initial_device());
            DEBUG_ACTION(printf("Current: team %d, thread %d \n", omp_get_team_num(), omp_get_thread_num());)
        }

        omp_target_free(d_alg, 0);
        R *vecmem_result = new R(size, &mr);
        vecmem_result->assign(map_result, map_result + size);
        return *vecmem_result;
    }

    template<class Algorithm,
            typename R = typename Algorithm::intermediate_result_t, typename T,
            typename... Rest>
    requires detail::is_mmap<Algorithm, T, Rest...>
            inline R &
    parallel_map(Algorithm& algorithm,
                 __attribute__((unused)) vecmem::memory_resource &mr,
                 vecpar::config config,
                 T &data,
                 Rest&... rest) {
        printf("***** OMPT library ***** \n");
        std::size_t size = data.size();

        value_type_t<T> *d_data = data.data();
     //   Algorithm *h_alg = new Algorithm();
      //  Algorithm *d_alg = (Algorithm*) omp_target_alloc(sizeof(Algorithm), omp_get_default_device());
      //  int cpy = omp_target_memcpy(d_alg, h_alg, sizeof(Algorithm), 0, 0, omp_get_default_device(), omp_get_initial_device()) ;// is_device_ptr(d_alg)
            //   Algorithm* d_alg = (Algorithm*) omp_alloc(sizeof(Algorithm), llvm_omp_target_shared_mem_alloc);
        //
     /*
        auto fn_copy = [&]<typename... P>(P& ...obj)
               -> std::tuple<std::conditional_t<
                    jagged_view<P>,P,P*>...> {
            return {([&](P &i) {
               P* d = (P*) omp_target_alloc(sizeof(P), 0);
               omp_target_memcpy(&d, &i, sizeof(P), 0, 0, omp_get_default_device(), omp_get_initial_device());
               return d;
            }(obj))...};
        };

        auto cleanup = [&]<typename... P>(P*...obj)
                -> void {
            return {([&](P* i) {
                omp_target_free(i, omp_get_default_device());
            }(obj))...};
        };

        auto d_ptrs = fn_copy(rest...);
*/
        size_t gs = config.m_gridSize;
        size_t bs = config.m_blockSize;

     //   typedef value_type_t<R>& (Algorithm::*funct_t) (value_type_t<T>&, Rest&...) const;
   //     funct_t f = &Algorithm::map;
      //  funct_t* d_map = (funct_t*)omp_target_alloc(sizeof(funct_t), omp_get_default_device());
     //   int res = omp_target_associate_ptr(&f, d_map, sizeof(f), 0, omp_get_default_device());
     //   printf("is mapped: %d\n", res);

     Algorithm *algo_ptr = new Algorithm();

#pragma omp target teams distribute parallel for \
    map(tofrom:d_data[0:size]) map(to:algo_ptr[0:sizeof(Algorithm)])\
    num_teams(gs) num_threads(bs) //is_device_ptr(d_map)
        for (size_t i = 0; i < size; i++) {
           // printf("block size = %d\n", bs);
            //      printf("%d %f \n", d_data[i], map_result[i]);
            algo_ptr->map(d_data[i], rest...);
      //      printf("Running on device? = %d\n", !omp_is_initial_device());
         //   DEBUG_ACTION(printf("Running on device? = %d\n", !omp_is_initial_device());)
            DEBUG_ACTION(printf("Current: team %d, thread %d \n", omp_get_team_num(), omp_get_thread_num());)
        }

    //    omp_target_free(d_alg, omp_get_default_device());
  //      omp_target_disassociate_ptr(&f, omp_get_default_device());
      //  std::apply(cleanup, d_ptrs);
        data.assign(d_data, d_data + size);
        return data;
    }

    template<class Algorithm,
            typename R = typename Algorithm::intermediate_result_t, typename T,
            typename... Rest>
    requires detail::is_mmap<Algorithm, T, Rest...> R &
    parallel_map(__attribute__((unused)) Algorithm &algorithm,
                 __attribute__((unused))  vecmem::memory_resource &mr,
                 T &data,
                 Rest&... rest) {
        printf("***** OMPT library ***** \n");
        std::size_t size = data.size();

        value_type_t<T> *d_data = data.data();
        Algorithm *d_alg = (Algorithm *) omp_target_alloc(sizeof(Algorithm), 0);

#pragma omp target teams distribute parallel for \
    map(tofrom:d_data[0:size]) is_device_ptr(d_alg)
        for (size_t i = 0; i < size; i++) {
            //      printf("%d %f \n", d_data[i], map_result[i]);
            d_alg->map(d_data[i], rest...);
           //algorithm.map(d_data[i], rest...);
            printf("Running on device? = %d\n", !omp_is_initial_device());
         //   DEBUG_ACTION(printf("Running on device? = %d\n", !omp_is_initial_device());)
            DEBUG_ACTION(printf("Current: team %d, thread %d \n", omp_get_team_num(), omp_get_thread_num());)
        }

        omp_target_free(d_alg, 0);

        data.assign(d_data, d_data + size);
        return data;
    }
}
#endif //VECPAR_OMPT_PARALLELIZATION_HPP
