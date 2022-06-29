#ifndef VECPAR_ALG3_HPP
#define VECPAR_ALG3_HPP

#include <vecmem/memory/memory_resource.hpp>

#include "vecpar/core/algorithms/parallelizable_map_filter.hpp"
#include "vecpar/core/definitions/config.hpp"

#include "../data_types.hpp"

class test_algorithm_3 : public vecpar::algorithm::parallelizable_map_filter_1<
                             vecmem::vector<double>, vecmem::vector<int>> {

public:
  TARGET test_algorithm_3(vecmem::memory_resource &mr)
      : parallelizable_map_filter_1(), m_mr(mr) {}

  TARGET double &map(double &result_i, const int &data_i) override {
    result_i = data_i * 1.0;
    return result_i;
  }

  TARGET bool filter(double &result_i) override {
    return (int(result_i) % 2 == 0);
  };

  vecmem::vector<double> &operator()(vecmem::vector<int> data,
                                     vecmem::vector<double> &result) {
    vecmem::vector<double> result_tmp(data.size(), &m_mr);
    int idx = 0;
    for (size_t i = 0; i < data.size(); i++) {
      map(result_tmp[i], data[i]);
      if (filter(result_tmp[i])) {
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