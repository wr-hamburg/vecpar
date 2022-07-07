#ifndef VECPAR_CLEANUP_HPP
#define VECPAR_CLEANUP_HPP

#include <gtest/gtest.h>
#include <vecmem/containers/vector.hpp>

namespace cleanup {

template <typename T> static void free(vecmem::vector<T> vec) {
  vec.clear();
  vec.shrink_to_fit();
}
}; // namespace cleanup

#endif // VECPAR_CLEANUP_HPP
