#ifndef VECPAR_TEST_ALGORITHM_7_HPP
#define VECPAR_TEST_ALGORITHM_7_HPP

#include <vecmem/containers/vector.hpp>
#include <vecmem/memory/memory_resource.hpp>

#include "vecpar/core/algorithms/parallelizable_map_reduce.hpp"
#include "vecpar/core/definitions/config.hpp"

#include "algorithm.hpp"

class test_algorithm_7 : public vecpar::algorithm::parallelizable_map_reduce<
                             Three,
                             /* result of reduction*/
                             double,
                             /* result of map*/
                             vecmem::vector<double>,
                             /* input collections */
                             vecmem::vector<double>, vecmem::vector<int>,
                             vecmem::jagged_vector<float>,
                             /* other input params */
                             float> {

public:
  TARGET test_algorithm_7() : parallelizable_map_reduce() {}

  TARGET double &map(double &result, const double &xi, const int &yi,
                     const vecmem::vector<float> &zi, float &a) const override {
    result = a * xi + yi * zi[0];
    return result;
  }

  TARGET double &map(double &result, const double &xi, const int &yi, auto zi,
                     float &a) const {
    result = a * xi + yi * zi[0];
    return result;
  }

  TARGET double *reduce(double *result, double &partial_result) const override {
    *result += partial_result;
    return result;
  }
};

#endif // VECPAR_TEST_ALGORITHM_7_HPP
