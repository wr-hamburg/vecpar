#ifndef VECPAR_MAP_REDUCE_HPP
#define VECPAR_MAP_REDUCE_HPP

#include "vecpar/core/algorithms/detail/map.hpp"
#include "vecpar/core/algorithms/detail/reduce.hpp"

namespace vecpar::algorithm {

    template<typename R, typename T, typename... Arguments>
    struct parallelizable_map_reduce : public vecpar::detail::parallel_map<R, T, Arguments...>,
                                       public vecpar::detail::parallel_reduce<R> {
        using result_type = R;
        using input_type = vecmem::vector<T>;
        using output_type_t = R;

        TARGET virtual R& map(R& result_i, T& input_i, Arguments... args) = 0;
        TARGET virtual R* reduce(R* result, R& partial_result) = 0;

    };

    template<typename TT, typename... Arguments>
    struct parallelizable_mmap_reduce : public vecpar::detail::parallel_mmap<TT, Arguments...>,
                                        public vecpar::detail::parallel_reduce<TT> {
        using result_type = TT;
        using input_type = vecmem::vector<TT>;
        using output_type_t = TT;

        TARGET virtual TT& map(TT& input_output_i, Arguments... args) = 0;
        TARGET virtual TT* reduce(TT* result, TT& partial_result) = 0;

    };
}
#endif //VECPAR_MAP_REDUCE_HPP
