#ifndef VECPAR_MAP_FILTER_HPP
#define VECPAR_MAP_FILTER_HPP

#include "vecpar/core/algorithms/detail/map.hpp"
#include "vecpar/core/algorithms/detail/filter.hpp"

namespace vecpar::algorithm {

    template<typename R, typename T, typename... Arguments>
    struct parallelizable_map_filter : public vecpar::detail::parallel_map<R, T, Arguments...>,
                                       public vecpar::detail::parallel_filter<R> {
        using result_type = R;

        TARGET virtual R& map(R& result_i, T& input_i, Arguments... args) = 0;
        TARGET virtual bool filter(R& partial_result) = 0;
    };

    /// partial specialization
    template<typename TT, typename... Arguments>
    struct parallelizable_map_filter<TT, TT, Arguments...>
            : public vecpar::detail::parallel_mmap<TT, TT, Arguments...>,
              public vecpar::detail::parallel_reduce<TT> {
        using result_type = TT;

        TARGET virtual TT& map(TT& input_output_i, Arguments... args) = 0;
        TARGET virtual bool filter(TT& partial_result) = 0;
    };
}
#endif //VECPAR_MAP_FILTER_HPP
