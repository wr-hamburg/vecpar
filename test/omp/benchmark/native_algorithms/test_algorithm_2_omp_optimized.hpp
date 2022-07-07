#ifndef VECPAR_ALG2_OMP_OPT_HPP
#define VECPAR_ALG2_OMP_OPT_HPP

#include "vecpar/core/algorithms/parallelizable_map_reduce.hpp"
#include "vecpar/core/definitions/config.hpp"

#include "../../../common/algorithm/algorithm.hpp"
#include "../../../common/data_types.hpp"

class test_algorithm_2_omp_optimized
    : public traccc::algorithm<double(vecmem::vector<int>, X)> {

public:
  test_algorithm_2_omp_optimized(vecmem::memory_resource &mr)
      : algorithm(), m_mr(mr) {}

  double operator()(vecmem::vector<int> data, X x, double &result) {
#pragma omp parallel for reduction(+ : result)
    for (int i = 0; i < data.size(); i++) {
      result += data[i] * x.f();
    }
    return result;
  }

  double operator()(vecmem::vector<int> data, X more_data) override {
    double result = 0; // new double();
    this->operator()(data, more_data, result);
    return result;
  }

private:
  vecmem::memory_resource &m_mr;
};

#endif