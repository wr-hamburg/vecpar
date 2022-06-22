#ifndef VECPAR_P_FILTER_HPP
#define VECPAR_P_FILTER_HPP

#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/containers/vector.hpp>

#include "vecpar/core/algorithms/parallelizable_filter.hpp"
#include "vecpar/core/definitions/config.hpp"

#include "vecpar/omp/detail/internal.hpp"

namespace vecpar::omp {

    template <typename Algorithm, typename T>
    vecmem::vector<T> &parallel_filter(Algorithm algorithm,
                                       vecmem::memory_resource &mr,
                                       vecmem::vector<T> &data) {

        vecmem::vector<T> *result = new vecmem::vector<T>(data.size(), &mr);
        internal::offload_filter(
                data.size(), result,
                [&](int idx, int &result_index, vecmem::vector<T> &local_result) mutable {
                    if (algorithm.filter(data[idx])) {
                        local_result[result_index] = data[idx];
                        result_index++;
                    }
                });
        return *result;
    }
}
#endif //VECPAR_P_FILTER_HPP
