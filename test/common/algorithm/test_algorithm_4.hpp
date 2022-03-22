#ifndef VECPAR_ALG4_HPP
#define VECPAR_ALG4_HPP

#include "vecpar/core/algorithms/parallelizable_map_reduce.hpp"
#include "vecpar/core/definitions/config.hpp"

#include "../data_types.hpp"
#include "algorithm.hpp"

class test_algorithm_4 :
        public traccc::algorithm<double*(vecmem::vector<double>)>,
        public vecpar::algorithm::parallelizable_map_reduce<double, double> {

public:

    TARGET test_algorithm_4(vecmem::memory_resource& mr) : parallelizable_map_reduce(), algorithm(), m_mr(mr) {}

    TARGET double& map(double& i) override {
        i = i * 2;
        return i;
    }

    TARGET double* reduce(double* result, double& result_i) override {
        *result += result_i;
        return result;
    }

    TARGET double* operator() (vecmem::vector<double> data, double* result) {
        for (int i = 0; i < data.size(); i++)
            reduce(result, map(data[i]));
        return result;
    }

    TARGET double* operator() (vecmem::vector<double> data) override {
        double* result = new double();
        this->operator()(data, result);
        return result;
    }

private:
    vecmem::memory_resource& m_mr;
};

#endif