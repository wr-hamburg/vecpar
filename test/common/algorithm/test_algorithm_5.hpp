#ifndef VECPAR_ALG5_HPP
#define VECPAR_ALG5_HPP

#include <vecmem/memory/memory_resource.hpp>

#include "vecpar/core/algorithms/parallelizable_map.hpp"
#include "vecpar/core/definitions/config.hpp"

#include "../data_types.hpp"
#include "algorithm.hpp"

class test_algorithm_5
    : public traccc::algorithm<vecmem::vector<double>(vecmem::vector<double>,
                                                      X)>,
      public vecpar::algorithm::parallelizable_mmap<
          vecpar::collection::count::One, vecmem::vector<double>, X> {

public:
  TARGET test_algorithm_5() : algorithm(), parallelizable_mmap() {}

  TARGET double &map(double &i, X &second_i) override {
    i = i + second_i.f();
    return i;
  }

  vecmem::vector<double> operator()(vecmem::vector<double> &data,
                                    X &more_data) override {
    for (size_t i = 0; i < data.size(); i++)
      map(data[i], more_data);
    return data;
  }
};

#endif