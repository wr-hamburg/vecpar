#ifndef VECPAR_PARALLELIZABLE_MAP_HPP
#define VECPAR_PARALLELIZABLE_MAP_HPP

#include "vecpar/core/algorithms/detail/map.hpp"

namespace vecpar::algorithm {

    template<typename R, typename T, typename... Arguments>
    struct parallelizable_map : public vecpar::detail::parallel_map<R, T, Arguments...> {
        using result_t = R;
        using input_t = T;
        using output_type_t = vecmem::vector<R>;
        using input_type = vecmem::vector<T>;

        TARGET virtual R& map(R& result_i, T& input_i, Arguments... args) = 0;

    };

    template <typename TT, typename... Arguments>
    struct parallelizable_mmap : vecpar::detail::parallel_mmap<TT, Arguments...> {
        using result_t = TT;
        using input_t = TT;
        using output_type_t = vecmem::vector<TT>;
        using input_type = vecmem::vector<TT>;

        TARGET virtual TT& map(TT& input_output_i, Arguments...) = 0;
    };

}
#endif //VECPAR_PARALLELIZABLE_MAP_HPP
