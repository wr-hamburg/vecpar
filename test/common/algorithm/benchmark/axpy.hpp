#ifndef VECPAR_AXPY_HPP
#define VECPAR_AXPY_HPP

#include <vecmem/containers/vector.hpp>
#include "vecpar/core/algorithms/parallelizable_map.hpp"

template <typename T>
class axpy :
        public vecpar::algorithm::parallelizable_mmap<
                Two, vecmem::vector<T>, vecmem::vector<T>, T> {
public:
    TARGET T &mapping_function(T &yi, const T &xi, T &a) const {
        yi = a * xi + yi;
        return yi;
    }
};

#endif //VECPAR_AXPY_HPP
