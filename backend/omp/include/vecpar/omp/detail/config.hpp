#ifndef VECPAR_OMP_CONFIG_HPP
#define VECPAR_OMP_CONFIG_HPP

#include "vecpar/core/definitions/config.hpp"

namespace vecpar::omp {

    static config getDefaultConfig(int size) {
        return {1, 16}; // TODO: retrieve these from hwloc
    }
}
#endif //VECPAR_OMP_CONFIG_HPP
