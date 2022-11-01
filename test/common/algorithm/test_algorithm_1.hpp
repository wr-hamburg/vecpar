#ifndef VECPAR_ALG1_HPP
#define VECPAR_ALG1_HPP

#include <vecmem/memory/memory_resource.hpp>
#include <omp.h>

#include "vecpar/core/algorithms/parallelizable_map_reduce.hpp"
#include "vecpar/core/definitions/config.hpp"

#include "../data_types.hpp"

class test_algorithm_1 : public vecpar::algorithm::parallelizable_map_reduce<
                             vecpar::collection::One, double,
                             vecmem::vector<double>, vecmem::vector<int>> {

public:
  TARGET test_algorithm_1()
      : parallelizable_map_reduce(){}

  TARGET double &map(double &result_i, const int &data_i) const  {
    result_i = data_i * 1.0;
#ifdef _OPENMP
    DEBUG_ACTION(printf("Running on device? = %d\n", !omp_is_initial_device());)
#endif
    return result_i;
  }

  TARGET double *reduce(double *result, double &result_i) const override {
    // printf("%f + %f \n ", *result, result_i);
    *result += result_i;
    return result;
  }

  double *operator()(vecmem::vector<int> data, double *result) {
    vecmem::vector<double> result_tmp(data.size());
    for (size_t i = 0; i < data.size(); i++)
      reduce(result, map(result_tmp[i], data[i]));
    return result;
  }

  double *operator()(vecmem::vector<int> data) {
    double *result = new double();
    this->operator()(data, result);
    return result;
  }

};
#endif // VECPAR_ALG1_HPP