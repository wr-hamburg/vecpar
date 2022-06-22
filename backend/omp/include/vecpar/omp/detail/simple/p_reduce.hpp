#ifndef VECPAR_P_REDUCE_HPP
#define VECPAR_P_REDUCE_HPP

#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/containers/vector.hpp>

#include "vecpar/core/algorithms/parallelizable_reduce.hpp"
#include "vecpar/core/definitions/config.hpp"

#include "vecpar/omp/detail/internal.hpp"

namespace vecpar::omp {

    /// generic function
    template <typename Function, typename... Arguments>
    void parallel_reduce(size_t size, Function f, Arguments... args) {
        internal::offload_reduce(size, f, args...);
    }

    /// implementation based on vecpar algorithms
    template <typename Algorithm, typename R>
    R &parallel_reduce(Algorithm algorithm,
                       __attribute__((unused)) vecmem::memory_resource &mr,
                       vecmem::vector<R> &data) {
        R *result = new R();
        internal::offload_reduce(
                data.size(), result,
                [&](R *r, R tmp) mutable { algorithm.reduce(r, tmp); }, data);

        return *result;
    }
}
#endif //VECPAR_P_REDUCE_HPP
