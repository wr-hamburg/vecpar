#ifndef VECPAR_FILTER_HPP
#define VECPAR_FILTER_HPP

#include "vecpar/core/definitions/common.hpp"

namespace vecpar::detail {

    template<typename T>
    struct parallel_filter {
        TARGET virtual bool filter(T&) = 0;
    };
}

#endif //VECPAR_FILTER_HPP
