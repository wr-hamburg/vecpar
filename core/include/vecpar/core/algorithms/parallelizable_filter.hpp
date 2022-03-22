#ifndef VECPAR_PARALLELIZABLE_FILTER_HPP
#define VECPAR_PARALLELIZABLE_FILTER_HPP

#include "vecpar/core/algorithms/detail/filter.hpp"

namespace vecpar::algorithm {

    template<typename R>
    struct parallelizable_filter : public vecpar::detail::parallel_filter<R> {
        using result_type = R;

        TARGET virtual bool filter(R& partial_result) = 0;
    };
}
#endif //VECPAR_PARALLELIZABLE_FILTER_HPP
