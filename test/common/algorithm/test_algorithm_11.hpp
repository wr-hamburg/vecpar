#ifndef VECPAR_TEST_ALGORITHM_11_HPP
#define VECPAR_TEST_ALGORITHM_11_HPP

#include <vecmem/containers/vector.hpp>
#include <vecmem/memory/memory_resource.hpp>

#include "vecpar/core/algorithms/parallelizable_map.hpp"
#include "vecpar/core/definitions/config.hpp"

#include "algorithm.hpp"

class test_algorithm_11 : public vecpar::algorithm::parallelizable_mmap<
                              One,
                              /* input collection */
                              vecmem::jagged_vector<double>> {

public:
  TARGET test_algorithm_11() : parallelizable_mmap() {}

  TARGET vecmem::vector<double> &map(vecmem::vector<double> &x) const override {
    for (int i = 0; i < static_cast<int>(x.size()); i++)
      x[i] += 5.0;
    return x;
  }

  TARGET auto map(auto x) const {
    for (int i = 0; i < static_cast<int>(x.size()); i++)
      x[i] += 5.0;
    return x;
  }
};
#endif // VECPAR_TEST_ALGORITHM_11_HPP
