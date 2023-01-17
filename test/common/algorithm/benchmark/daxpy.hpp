#ifndef VECPAR_DAXPY_HPP
#define VECPAR_DAXPY_HPP

#include <vecmem/containers/vector.hpp>
#include "vecpar/core/algorithms/parallelizable_map.hpp"

class daxpy :
        public vecpar::algorithm::parallelizable_mmap<
                  Two, vecmem::vector<double>, vecmem::vector<double>, double> {

public:
    TARGET daxpy() : parallelizable_mmap() {}

    TARGET double &mapping_function(double &yi, const double &xi, double &a) const {
        yi = a * xi + yi;
        return yi;
    }
};

#endif //VECPAR_DAXPY_HPP
