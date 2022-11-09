#ifndef VECPAR_OMPT_HPP
#define VECPAR_OMPT_HPP

#include <vecmem/memory/host_memory_resource.hpp>
#include "vecpar/omp/omp_parallelization.hpp"
#include "vecpar/ompt/ompt_parallelization.hpp"

namespace ompt {

    /// only parallelizable_map/mmap with one vecmem::vector has OpenMP Target support compiled with GCC
    template<typename Algorithm, typename R, typename T, typename... All>
    constexpr bool supports_ompt() {
//#if defined(__GNU__)
        return
                (vecpar::detail::is_mmap_1<Algorithm, R, All...> ||
                 vecpar::detail::is_map_1<Algorithm, R, T, All...>)
                &&
                (vecpar::collection::Vector_type<T> &&
                 !vecpar::collection::Jagged_vector_type<T>);
//#endif
        //  return false;
    }

    template<class MemoryResource, class Algorithm,
            class R = typename Algorithm::intermediate_result_t, class T,
            typename... Arguments>
    requires vecpar::algorithm::is_map<Algorithm, R, T, Arguments...> ||
             vecpar::algorithm::is_mmap<Algorithm, R, Arguments...>
    R &parallel_algorithm(Algorithm algorithm, MemoryResource &mr,
                          vecpar::config config, T &data,
                          Arguments &...args) {

        if constexpr(supports_ompt<Algorithm, R, T, Arguments...>()) {
            if (omp_get_num_devices() > 0 &&
                std::is_base_of<vecmem::host_memory_resource, MemoryResource>::value) {
                printf("Dispatch to ompt library!!!\n");
                return vecpar::ompt::parallel_map<Algorithm, R, T, Arguments...>(
                        algorithm, mr, config, data, args...);
            }
        }

        return vecpar::omp::parallel_map<Algorithm, R, T, Arguments...>(
                algorithm, mr, config, data, args...);
    }

    template<class MemoryResource, class Algorithm,
            class R = typename Algorithm::intermediate_result_t, class T,
            typename... Arguments>
    requires vecpar::algorithm::is_map<Algorithm, R, T, Arguments...> ||
             vecpar::algorithm::is_mmap<Algorithm, R, Arguments...>
    R &parallel_algorithm(Algorithm algorithm, MemoryResource &mr, T &data,
                          Arguments &...args) {

        if constexpr(supports_ompt<Algorithm, R, T, Arguments...>()) {
            if (omp_get_num_devices() > 0 &&
                std::is_base_of<vecmem::host_memory_resource, MemoryResource>::value) {
                printf("Dispatch to ompt library!!!\n");
                return vecpar::ompt::parallel_map<Algorithm, R, T, Arguments...>(
                        algorithm, mr, data, args...);
            }
        }

        return vecpar::omp::parallel_map<Algorithm, R, T, Arguments...>(
                algorithm, mr, data, args...);
    }

    template<class MemoryResource, class Algorithm,
            class R = typename Algorithm::intermediate_result_t, class T,
            typename... Arguments>
    R &parallel_algorithm(__attribute__((unused))  Algorithm algorithm,
                          __attribute__((unused))  MemoryResource &mr,
                          __attribute__((unused))  vecpar::config config,
                          __attribute__((unused))  T &data,
                          __attribute__((unused))  Arguments &...args) {
        throw std::logic_error("Not implemented yet");
    }
}
#endif //VECPAR_OMPT_HPP
