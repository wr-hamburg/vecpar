#ifndef VECPAR_ALG4_HPP
#define VECPAR_ALG4_HPP

#include <vecmem/memory/memory_resource.hpp>

#include "vecpar/core/algorithms/parallelizable_map_reduce.hpp"
#include "vecpar/core/definitions/config.hpp"

#include "../data_types.hpp"
#include "algorithm.hpp"

class test_algorithm_4
    : public traccc::algorithm<double *(vecmem::vector<double>)>,
      public vecpar::algorithm::parallelizable_mmap_reduce<
          vecpar::collection::One, double, vecmem::vector<double>> {

public:
  TARGET test_algorithm_4() : algorithm(), parallelizable_mmap_reduce() {}

  TARGET double &mapping_function(double &i) const {
    i = i * 2;
    return i;
  }

  TARGET double *reducing_function(double *result, double &result_i) const {
    *result += result_i;
    return result;
  }

  double *operator()(vecmem::vector<double> &data, double *result) {
    for (size_t i = 0; i < data.size(); i++)
      reducing_function(result, mapping_function(data[i]));
    return result;
  }

  double *operator()(vecmem::vector<double> &data) override {
    double *result = new double();
    this->operator()(data, result);
    return result;
  }
};

#endif