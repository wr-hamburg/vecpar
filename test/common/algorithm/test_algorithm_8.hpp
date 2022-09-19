#ifndef VECPAR_TEST_ALGORITHM_8_HPP
#define VECPAR_TEST_ALGORITHM_8_HPP

#include <cmath>
#include <vecmem/containers/vector.hpp>
#include <vecmem/memory/memory_resource.hpp>

#include "vecpar/core/algorithms/parallelizable_map_filter.hpp"
#include "vecpar/core/definitions/config.hpp"

#include "algorithm.hpp"

class test_algorithm_8
    : public vecpar::algorithm::parallelizable_map_filter<
          Four,
          /* result of map*/
          vecmem::vector<double>,
          /* input collections */
          vecmem::vector<double>, vecmem::vector<int>, vecmem::vector<float>,
          vecmem::jagged_vector<float>,
          /* other input params */
          float> {

public:
  TARGET test_algorithm_8() : parallelizable_map_filter() {}

  TARGET double &map(double &result, const double &xi, const int &yi,
                     const float &zi, const vecmem::vector<float> &ti,
                     float &a) override {
    result = a * xi + yi * zi * ti[0];
    return result;
  }

  TARGET double &map(double &result, const double &xi, const int &yi,
                     const float &zi, auto ti, float &a) {
    result = a * xi + yi * zi * ti[0];
    return result;
  }

  // keep only the negative numbers
  TARGET bool filter(double &item) override { return (item < 0); }
};

#endif // VECPAR_TEST_ALGORITHM_8_HPP
