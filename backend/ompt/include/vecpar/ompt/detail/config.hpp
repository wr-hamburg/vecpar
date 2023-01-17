#ifndef VECPAR_OMPT_CONFIG_HPP
#define VECPAR_OMPT_CONFIG_HPP

#include "vecpar/core/definitions/config.hpp"

namespace vecpar::ompt {

static inline config getDefaultConfig() {
  return vecpar::config(); // let the OMP runtime decide based on hardware
}

} // namespace vecpar::ompt
#endif // VECPAR_OMPT_CONFIG_HPP
