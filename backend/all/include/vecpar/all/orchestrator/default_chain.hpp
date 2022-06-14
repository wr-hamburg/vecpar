#ifndef VECPAR_DEFAULT_CHAIN_HPP
#define VECPAR_DEFAULT_CHAIN_HPP

#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/containers/vector.hpp>
#include "common.hpp"

namespace vecpar {

    //TODO: have this compiler restriction in vecpar::algorithms too
    //   template<typename> typename S = vecmem::vector,
    template<class MemoryResource,
            typename R, typename H, typename... OtherInput>
    class chain {

    public:

        chain(MemoryResource& mem) : m_mr(mem) {}

        chain& with_config(vecpar::config config) {
            m_config = config;
            return *this;
        }

        template<typename... Args>
        chain& with_algorithms(Args... args) {
            /// cannot call with_algorithms more than once
            assertm(!algorithms_set, ALGORITHMS_ALREADY_SET);

            composition = compose(wrapper(args)...);
            algorithms_set = true;

            return *this;
        }

        R execute(H& coll, OtherInput... rest) {
            /// cannot invoke chain execution without providing algorithms
            assertm(algorithms_set, MISSING_ALGORITHMS);

            return composition(coll, rest...);
        }

    private:
        template <class Algorithm,
                class Input = typename Algorithm::input_type,
                class Output = typename Algorithm::output_type_t,
                class input_t = typename Algorithm::input_t,
                class result_t = typename Algorithm::result_t> //,
        auto wrapper(Algorithm& algorithm) {
            return [&](Input& coll, OtherInput... otherInput) -> Output& {
                if constexpr (std::is_base_of<vecpar::detail::parallel_map<result_t, input_t, OtherInput...>, Algorithm>::value ||
                              std::is_base_of<vecpar::detail::parallel_mmap<result_t, OtherInput...>, Algorithm>::value) {
                   return vecpar::parallel_algorithm(algorithm, m_mr, m_config, coll, otherInput...);
                 }
                else {
                    return vecpar::parallel_algorithm(algorithm, m_mr, m_config, coll);
                }
            };
        }

    protected:
        MemoryResource& m_mr;
        vecpar::config m_config;
        std::function<R(H&, OtherInput...)> composition;
        bool algorithms_set = false;
    };
}

#endif //VECPAR_DEFAULT_CHAIN_HPP