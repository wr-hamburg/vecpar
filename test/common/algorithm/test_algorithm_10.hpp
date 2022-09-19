#ifndef VECPAR_TEST_ALGORITHM_10_HPP
#define VECPAR_TEST_ALGORITHM_10_HPP

#include <cmath>
#include <vecmem/containers/vector.hpp>
#include <vecmem/memory/memory_resource.hpp>

#include "vecpar/core/algorithms/parallelizable_map.hpp"
#include "vecpar/core/definitions/config.hpp"

#include "algorithm.hpp"

class test_algorithm_10
    : public vecpar::algorithm::parallelizable_mmap<
          Five,
          /* input collections */
          vecmem::jagged_vector<double>, vecmem::jagged_vector<double>,
          vecmem::vector<int>, vecmem::vector<int>, vecmem::jagged_vector<int>,
          /* other input params */
          double> {

public:
  TARGET test_algorithm_10() : parallelizable_mmap() {}

  TARGET vecmem::vector<double> &
  map(vecmem::vector<double> &x, const vecmem::vector<double> &y, const int &z,
      const int &t, const vecmem::vector<int> &v, double &a) override {
    for (int i = 0; i < static_cast<int>(x.size()); i++)
      x[i] = a * y[i] + x[i] - z * t * v[i];
    return x;
  }

  TARGET auto map(auto x, auto y, const int &z, const int &t, const auto v,
                  double &a) {
    for (int i = 0; i < static_cast<int>(x.size()); i++)
      x[i] = a * y[i] + x[i] - z * t * v[i];
    return x;
  }
  /*
      TARGET vecmem::device_vector<double> map(vecmem::device_vector<double> x,
     const vecmem::device_vector<double> y, const int& z, const int& t, const
     vecmem::device_vector<int> v, double &a)  { for (int i = 0; i < x.size();
     i++) {
             // printf("x[%d] and y[%d] before update: %f %f \n", i, i, x[i],
     y[i]); x[i] = a * y[i] + x[i] - z * t * v[i];
           //   printf("x[%d] after update = %f \n", i, x[i]);
          }
          return x;
      }
      */
};
#endif // VECPAR_TEST_ALGORITHM_10_HPP
