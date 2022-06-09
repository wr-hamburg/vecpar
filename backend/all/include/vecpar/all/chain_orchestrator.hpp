#ifndef VECPAR_CHAIN_HPP
#define VECPAR_CHAIN_HPP

#include <utility>
#include <functional>

#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/containers/vector.hpp>

//TODO: extract this in a dependency module
namespace external {
#define INVOCABLE_ASSERTION_FAILED_ERROR_MESSAGE\
    "An object should be invokable with the provided arguments"

    template<typename F>
    auto compose(F &&f) {
        return [a = std::forward<F>(f)](auto &&... args)  {
            static_assert(std::is_invocable_v<F, decltype(args)...>,
                          INVOCABLE_ASSERTION_FAILED_ERROR_MESSAGE);
            return a(std::forward<decltype(args)>(args)...);
        };
    }

    template<typename F1, typename F2, typename... Fs>
    auto compose(F1 &&f1, F2 &&f2, Fs &&... fs) {
        return compose(
                [first = std::forward<F1>(f1), second = std::forward<F2>(f2)]
                        (auto &&... args)  {
                    static_assert(std::is_invocable_v<F1, decltype(args)...>,
                                  INVOCABLE_ASSERTION_FAILED_ERROR_MESSAGE
                    );
                    static_assert(
                            std::is_invocable_v<F2,
                                    std::result_of_t<F1(decltype(args)...)>
                            >,
                            INVOCABLE_ASSERTION_FAILED_ERROR_MESSAGE
                    );
                    return second(first(std::forward<decltype(args)>(args)...));
                },
                std::forward<Fs>(fs)...
        );
    }
}

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
            composition = external::compose(wrapper(args)...);
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

   typename std::enable_if_t<!std::is_base_of<vecmem::host_memory_resource, MemoryResource>::value, bool> = true>

        */

        //TODO: see if wrapper duplication can be avoided
        template <class Algorithm,
                class Input = typename Algorithm::input_type,
                class Output = typename Algorithm::output_type_t>
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