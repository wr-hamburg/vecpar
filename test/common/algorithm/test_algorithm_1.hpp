#ifndef VECPAR_ALG1_HPP
#define VECPAR_ALG1_HPP

#include "vecpar/core/algorithms/parallelizable_map_reduce.hpp"
#include "vecpar/core/definitions/config.hpp"

#include "../data_types.hpp"

class test_algorithm_1 :
    public vecpar::algorithm::parallelizable_map_reduce<double, int> {

public:
    TARGET test_algorithm_1(vecmem::memory_resource& mr) : parallelizable_map_reduce(), m_mr(mr) {}

    TARGET double& map(double& result_i, int& data_i) override {
        result_i = data_i * 1.0;
        return result_i;
    }

    TARGET double* reduce(double* result, double& result_i) override {
       // printf("%f + %f \n ", *result, result_i);
        *result += result_i;
        return result;
    }

    double* operator() (vecmem::vector<int> data, double* result) {
        vecmem::vector<double> result_tmp(data.size(), &m_mr);
        for (size_t i = 0; i < data.size(); i++)
          reduce(result, map(result_tmp[i], data[i]));
        return result;
    }

    double* operator() (vecmem::vector<int> data)  {
        double* result = new double();
        this->operator()(data, result);
        return result;
    }

private:
    vecmem::memory_resource& m_mr;
};

#endif // VECPAR_ALG1_HPP