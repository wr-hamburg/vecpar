#ifndef VECPAR_SAXPY_HPP
#define VECPAR_SAXPY_HPP

#include <vecmem/containers/vector.hpp>
#include "vecpar/core/algorithms/parallelizable_map.hpp"

class saxpy :
        public vecpar::algorithm::parallelizable_mmap<
                  Two, vecmem::vector<float>, vecmem::vector<float>, float> {

public:
    TARGET saxpy() : parallelizable_mmap() {}

    TARGET float &mapping_function(float &yi, const float &xi, float &a) const {
        yi = a * xi + yi;
        return yi;
    }
};

#endif //VECPAR_SAXPY_HPP
