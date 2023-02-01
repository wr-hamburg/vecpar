#ifndef VECPAR_ALG2_HPP
#define VECPAR_ALG2_HPP

#include <vecmem/memory/memory_resource.hpp>

#include "vecpar/core/algorithms/parallelizable_map_reduce.hpp"
#include "vecpar/core/definitions/config.hpp"

#include "../data_types.hpp"
#include "algorithm.hpp"

class test_algorithm_2
    :// public traccc::algorithm<double *(vecmem::vector<int>, X)>,
      public vecpar::algorithm::parallelizable_map_reduce<
          vecpar::collection::One, double, vecmem::vector<double>,
          vecmem::vector<int>, X> {

public:
  test_algorithm_2()
      : //algorithm(),
      parallelizable_map_reduce() {}

  TARGET double &mapping_function(double &result_i, const int &first_i,
                     X &second_i) const {
    result_i = first_i * second_i.f();
    return result_i;
  }

  TARGET double *reducing_function(double *result, double &result_i) const  {
    if (result_i > 0)
      *result += result_i;
    return result;
  }

  double *operator()(vecmem::vector<int> &data, X &x, double *result) {
    vecmem::vector<double> result_tmp(data.size());
    for (size_t i = 0; i < data.size(); i++)
      reducing_function(result, mapping_function(result_tmp[i], data[i], x));
    return result;
  }

  double *operator()(vecmem::vector<int> &data, X &more_data) { //override {
    double *result = new double();
    this->operator()(data, more_data, result);
    return result;
  }
};

#endif