#ifndef VECPAR_ALG2_SEQ_HPP
#define VECPAR_ALG2_SEQ_HPP

#include "vecpar/core/algorithms/parallelizable_map_reduce.hpp"
#include "vecpar/core/definitions/config.hpp"

#include "../../common/data_types.hpp"
#include "../../common/algorithm/algorithm.hpp"


class test_algorithm_2_seq :
        public traccc::algorithm<double(vecmem::vector<int>, X)>{

public:

    TARGET test_algorithm_2_seq(vecmem::memory_resource& mr): algorithm(), m_mr(mr) {}

    TARGET double operator() (vecmem::vector<int> data, X x, double& result) {
        for (int i = 0; i < data.size(); i++)
            result += data[i] * x.f();
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