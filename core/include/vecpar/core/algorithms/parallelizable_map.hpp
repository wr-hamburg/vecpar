#ifndef VECPAR_PARALLELIZABLE_MAP_HPP
#define VECPAR_PARALLELIZABLE_MAP_HPP

#include "vecpar/core/algorithms/detail/map.hpp"

namespace vecpar::algorithm {

    template<typename R, typename T, typename... Arguments>
    struct parallelizable_map : public vecpar::detail::parallel_map<R, T, Arguments...> {
        using result_type = R;

        TARGET virtual R& map(R& result_i, T& input_i, Arguments... args) = 0;

    };

    template <typename TT, typename... Arguments>
    struct parallelizable_mmap : vecpar::detail::parallel_mmap<TT, Arguments...> {
        using result_type = TT;

        TARGET virtual TT& map(TT& input_output_i, Arguments...) = 0;
    };

}
#endif //VECPAR_PARALLELIZABLE_MAP_HPP
