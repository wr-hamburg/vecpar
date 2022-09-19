#ifndef VECPAR_TEST_ALGORITHM_9_HPP
#define VECPAR_TEST_ALGORITHM_9_HPP

#include <cmath>
#include <vecmem/containers/vector.hpp>
#include <vecmem/memory/memory_resource.hpp>

#include "vecpar/core/algorithms/parallelizable_map_filter.hpp"
#include "vecpar/core/definitions/config.hpp"

#include "algorithm.hpp"

class test_algorithm_9
    : public vecpar::algorithm::parallelizable_mmap_filter<
          Five,
          /* input collections */
          vecmem::vector<double>, vecmem::vector<int>, vecmem::vector<float>,
          vecmem::vector<float>, vecmem::vector<int>,
          /* other input params */
          float> {

public:
  TARGET test_algorithm_9() : parallelizable_mmap_filter() {}

  TARGET double &map(double &xi, const int &yi, const float &zi,
                     const float &ti, const int &vi, float &a) const override {
    xi = a * xi + yi * zi * ti + vi;
    return xi;
  }

  // keep only the positive numbers
  TARGET bool filter(double &item) const override { return (item > 0); }
};

#endif // VECPAR_TEST_ALGORITHM_9_HPP
