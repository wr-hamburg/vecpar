#ifndef VECPAR_MAP_FILTER_HPP
#define VECPAR_MAP_FILTER_HPP

#include "vecpar/core/algorithms/detail/filter.hpp"
#include "vecpar/core/algorithms/detail/map.hpp"

namespace vecpar::algorithm {

    template <typename R, typename T, typename... Arguments>
    struct parallelizable_map_filter_1 :
            public vecpar::detail::parallel_map_1<R, T, Arguments...>,
            public vecpar::detail::parallel_filter<R> {
        using result_t = R;
    };

    template <typename R, typename... Arguments>
    struct parallelizable_mmap_filter_1 :
            public vecpar::detail::parallel_mmap_1<R, Arguments...>,
            public vecpar::detail::parallel_filter<R> {
        using result_t = R;
    };

} // namespace vecpar::algorithm
#endif // VECPAR_MAP_FILTER_HPP
