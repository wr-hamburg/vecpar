#ifndef VECPAR_ALG2_OMP_HPP
#define VECPAR_ALG2_OMP_HPP

#include "vecpar/core/algorithms/parallelizable_map_reduce.hpp"
#include "vecpar/core/definitions/config.hpp"

#include "../../common/data_types.hpp"
#include "../../common/algorithm/algorithm.hpp"

class test_algorithm_2_omp :
        public traccc::algorithm<double(vecmem::vector<int>, X)>{

public:

    TARGET test_algorithm_2_omp(vecmem::memory_resource& mr): algorithm(), m_mr(mr) {}

    TARGET double operator() (vecmem::vector<int> data, X x, double& result) {
        #pragma omp parallel for
        for (int i = 0; i < data.size(); i++) {
             double result_i = data[i] * x.f();
             #pragma omp critical
             {
                result += result_i;
             }
        }
        return result;
    }

    TARGET double operator() (vecmem::vector<int> data, X more_data) override {
        double result = 0;
        this->operator()(data, more_data, result);
        return result;
    }

private:
    vecmem::memory_resource& m_mr;
};

#endif