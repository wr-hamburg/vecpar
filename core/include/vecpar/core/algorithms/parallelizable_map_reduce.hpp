#ifndef VECPAR_MAP_REDUCE_HPP
#define VECPAR_MAP_REDUCE_HPP

#include "vecpar/core/algorithms/detail/map.hpp"
#include "vecpar/core/algorithms/detail/reduce.hpp"

namespace vecpar::algorithm {

    /// R is a collection of <Result>
    /// TODO: validate that at compile-time
    template <typename Result, typename R, typename T, typename... Arguments>
    struct parallelizable_map_reduce_1 :
            public vecpar::detail::parallel_map_1<R, T, Arguments...>,
            public vecpar::detail::parallel_reduce<R> {
        using input_t = T;
        using result_t = Result;
        using intermediate_result_t = R;
    };

    /// R is a collection of <Result>
    template <typename Result, typename R, typename... Arguments>
    struct parallelizable_mmap_reduce_1 :
            public vecpar::detail::parallel_mmap_1<R, Arguments...>,
            public vecpar::detail::parallel_reduce<R> {
        using input_t = R;
        using result_t = Result;
        using intermediate_result_t = R;
    };

} // namespace vecpar::algorithm
#endif // VECPAR_MAP_REDUCE_HPP
