#ifndef VECPAR_MAP_HPP
#define VECPAR_MAP_HPP

#include "vecpar/core/definitions/common.hpp"

// TODO: Templated on vecmem::vector or vecmem::jagged_vector
namespace vecpar::detail {

/**
 *  Input is vecmem::vector<T> and Args...
 *  Output is vecmem::vector<R>
 */
template <typename R, typename T, typename... Arguments> struct parallel_map {
  TARGET virtual R &map(R &, T &, Arguments...) = 0;
};

template <typename TT, typename... Arguments> struct parallel_mmap {
  TARGET virtual TT &map(TT &, Arguments...) = 0;
};

} // namespace vecpar::detail
#endif // VECPAR_MAP_HPP
