#ifndef VECPAR_ALG3_HPP
#define VECPAR_ALG3_HPP

#include <vecmem/memory/memory_resource.hpp>

#include "vecpar/core/algorithms/parallelizable_map_filter.hpp"
#include "vecpar/core/definitions/config.hpp"

#include "../data_types.hpp"

class test_algorithm_3 : public vecpar::algorithm::parallelizable_map_filter<
                             vecpar::collection::One, vecmem::vector<double>,
                             vecmem::vector<int>> {

public:
  TARGET test_algorithm_3(vecmem::memory_resource &mr)
      : parallelizable_map_filter(), m_mr(mr) {}

  TARGET double &mapping_function(double &result_i, const int &data_i) const {
    result_i = data_i * 1.0;
    return result_i;
  }

  TARGET bool filtering_function(double &result_i) const {
    return (int(result_i) % 2 == 0);
  };

  vecmem::vector<double> &operator()(vecmem::vector<int> data,
                                     vecmem::vector<double> &result) {
    vecmem::vector<double> result_tmp(data.size(), &m_mr);
    int idx = 0;
    for (size_t i = 0; i < data.size(); i++) {
      mapping_function(result_tmp[i], data[i]);
      if (filtering_function(result_tmp[i])) {
        result[idx] = result_tmp[i];
        idx++;
      }
    }
    result.resize(idx);
    return result;
  }

  vecmem::vector<double> &operator()(vecmem::vector<int> data) {
    vecmem::vector<double> *result =
        new vecmem::vector<double>(data.size(), &m_mr);
    this->operator()(data, *result);
    return *result;
  }

private:
  vecmem::memory_resource &m_mr;
};
#endif