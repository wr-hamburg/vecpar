#ifndef VECPAR_CCOMMON_HPP
#define VECPAR_CCOMMON_HPP

#include <functional>
#include <cassert>

#include "../../../../../../extern/composition/Compose.hpp"
#include "vecpar/core/definitions/config.hpp"

#define MISSING_ALGORITHMS\
    "A chain should have at least one algorithm."

#define ALGORITHMS_ALREADY_SET \
    "A list of algorithms was already provided."

#define assertm(exp, msg) assert(((void)msg, exp))

#endif //VECPAR_CCOMMON_HPP
