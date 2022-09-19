#ifndef VECPAR_TEST_ALGORITHM_6_HPP
#define VECPAR_TEST_ALGORITHM_6_HPP

#include <vecmem/containers/vector.hpp>
#include <vecmem/memory/memory_resource.hpp>

#include "vecpar/core/algorithms/parallelizable_map.hpp"
#include "vecpar/core/definitions/config.hpp"

#include "algorithm.hpp"

class test_algorithm_6
    : public traccc::algorithm<vecmem::vector<float>(
          vecmem::vector<float>, vecmem::vector<float>, float)>,
      public vecpar::algorithm::parallelizable_mmap<
          Two, vecmem::vector<float>, vecmem::vector<float>, float> {

public:
  TARGET test_algorithm_6() : algorithm(), parallelizable_mmap() {}

  TARGET float &map(float &yi, const float &xi, float &a) const override {
    yi = a * xi + yi;
    return yi;
  }

  vecmem::vector<float> operator()(vecmem::vector<float> &y,
                                   vecmem::vector<float> &x,
                                   float &a) override {
    for (size_t i = 0; i < y.size(); i++)
      map(y[i], x[i], a);
    return y;
  }
};

#endif // VECPAR_TEST_ALGORITHM_6_HPP
