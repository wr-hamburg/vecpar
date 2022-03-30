#ifndef VECPAR_OMP_CONFIG_HPP
#define VECPAR_OMP_CONFIG_HPP

#include "vecpar/core/definitions/config.hpp"

namespace vecpar::omp {

    static config getDefaultConfig(__attribute__((unused)) int size) {
        return {1, 16, 0}; // TODO: get from user or retrieve based on hardware; use size
    }
}
#endif //VECPAR_OMP_CONFIG_HPP
