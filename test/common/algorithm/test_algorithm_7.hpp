#ifndef VECPAR_TEST_ALGORITHM_7_HPP
#define VECPAR_TEST_ALGORITHM_7_HPP

#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/containers/vector.hpp>

#include "vecpar/core/algorithms/parallelizable_map.hpp"
#include "vecpar/core/definitions/config.hpp"

#include "algorithm.hpp"

class test_algorithm_7 :
          public vecpar::algorithm::parallelizable_map_reduce<double, double, int, float, float> {

public:
    TARGET test_algorithm_7() : parallelizable_map_reduce() {}

    TARGET double &map(double &result, double& xi, int yi, float zi, float a) override {
       result = a * xi + yi * zi;
       return result;
    }

    double *reduce(double *result, double &partial_result) override {
        *result += partial_result;
        return result;
    }
};


#endif //VECPAR_TEST_ALGORITHM_7_HPP
