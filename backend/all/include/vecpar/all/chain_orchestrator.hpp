#ifndef VECPAR_CHAIN_HPP
#define VECPAR_CHAIN_HPP

#include <functional>

#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/containers/vector.hpp>
#include "../../extern/composition/Compose.hpp"

namespace vecpar {

    static bool first_fn = false;

    //TODO: have this compiler restriction in vecpar::algorithms too
    template<class MemoryResource,
            typename R, typename H,
            template<typename> typename S = vecmem::vector,
            typename... OtherInput>
    class chain_orchestrator {

    public:

        chain_orchestrator(MemoryResource& mem) :
               m_mr(mem) {
            first_fn = true;
        }

        chain_orchestrator& with_config(vecpar::config config) {
            m_config = config;
            return *this;
        }

        template<typename... Args>
        chain_orchestrator& with_algorithms(Args... args) {
            composition = compose(wrapper(args)...);
            return *this;
        }

        R execute(H& coll, OtherInput... rest) {
            return composition(coll, rest...);
        }

    private:

        /*
        template <class Algorithm,
                class Input = typename Algorithm::input_type,
                class Output = typename Algorithm::output_type_t,
                typename std::enable_if_t<std::is_base_of<vecmem::host_memory_resource, MemoryResource>::value, bool> = true>
        auto wrapper(Algorithm& algorithm) {

#if defined(__CUDA__) && defined(__clang__)
            //
#else
            return [&](Input& coll, OtherInput... otherInput) -> Output& {
                if (first_fn) {
                    // pass extra input argument ONLY to the first function from the chain
                    first_fn = false;
                    return vecpar::parallel_algorithm(algorithm, m_mr, m_config, coll, otherInput...);
                }
                else
                    return vecpar::parallel_algorithm(algorithm, m_mr, m_config, coll);
            };
#endif
        }
*/
        //TODO: see if wrapper duplication can be avoided
        template <class Algorithm,
                class Input = typename Algorithm::input_type,
                class Output = typename Algorithm::output_type_t>//,
   //             typename std::enable_if_t<!std::is_base_of<vecmem::host_memory_resource, MemoryResource>::value, bool> = true>
        auto wrapper(Algorithm& algorithm) {
            return [&](Input& coll, OtherInput... otherInput) -> Output& {
                if (first_fn) {
                    // pass extra input argument ONLY to the first function from the chain
                    first_fn = false;
                    return vecpar::parallel_algorithm(algorithm, m_mr, m_config, coll, otherInput...);
                }
                else
                    return vecpar::parallel_algorithm(algorithm, m_mr, m_config, coll);
            };
        }

    private:
        MemoryResource& m_mr;
        vecpar::config m_config;
        std::function<R(H&, OtherInput...)> composition;
    };
}
#endif //VECPAR_CHAIN_HPP