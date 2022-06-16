#ifndef VECPAR_OMP_CONFIG_HPP
#define VECPAR_OMP_CONFIG_HPP

#include "vecpar/core/definitions/config.hpp"

namespace vecpar::omp {

static inline config getDefaultConfig() {
  return vecpar::config(); // let the OMP runtime decide based on hardware
}
} // namespace vecpar::omp
#endif // VECPAR_OMP_CONFIG_HPP
