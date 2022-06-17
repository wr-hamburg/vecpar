#include <gtest/gtest.h>

#include <cuda.h>

#include <vecmem/containers/vector.hpp>
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/host_memory_resource.hpp>

#include "../../common/infrastructure/TimeTest.hpp"
#include "../../common/infrastructure/sizes.hpp"

#include "../../common/algorithm/test_algorithm_1.hpp"
#include "../../common/algorithm/test_algorithm_2.hpp"
#include "../../common/algorithm/test_algorithm_3.hpp"
#include "../../common/algorithm/test_algorithm_4.hpp"
#include "../../common/algorithm/test_algorithm_5.hpp"

#include "vecpar/cuda/detail/generic/raw_data.hpp"

#include <functional>

using namespace std::placeholders;

namespace {

class ChainGpuTest : public TimeTest, public testing::WithParamInterface<int> {
public:
  ChainGpuTest() {
    vec = new vecmem::vector<int>(GetParam(), &mr);
    vec_d = new vecmem::vector<double>(GetParam(), &mr);
    for (int i = 0; i < vec->size(); i++) {
      vec->at(i) = i;
      vec_d->at(i) = i * 1.0;
      expectedReduceResult += i * 1.0;
      expectedFilterReduceResult += (i % 2 == 0) ? (i * 1.0) * 2 : 0;
    }
    printf("*******************************\n");
  }

  ~ChainGpuTest() {
    //      free(vec);
    //    free(vec_d);
  }

protected:
  vecmem::host_memory_resource mr;
  vecmem::cuda::device_memory_resource d_mem;
  vecmem::vector<int> *vec;
  vecmem::vector<double> *vec_d;
  double expectedReduceResult = 0;
  double expectedFilterReduceResult = 0;
};

/*
TEST_P(ChainGpuTest, Parallel_Map_Map_Chained) {
    test_algorithm_3 first_alg(mr);
    test_algorithm_4 second_alg;
//     vecpar::config c{static_cast<int>(vec->size()/64 + 1), 64};

    double result = 0;
    vecpar::cuda::chain(result, *vec, first_alg, second_alg);

    printf("Result after chain execution: %f\n", result);
}
*/
/*
    TEST_P(ChainGpuTest, Parallel_Map_Map_Chained) {
        test_algorithm_3 first_alg(mr);
        test_algorithm_4 second_alg;

        vecpar::config c{static_cast<int>(vec->size()/64 + 1), 64};

        // copy input data from host to device
        auto data_buffer = copy.to(vecmem::get_data(*vec), d_mem,
   vecmem::copy::type::host_to_device); auto data_view =
   vecmem::get_data(data_buffer);

        // allocate result on host and device
        vecmem::vector<double> result (vec->size(), &mr);
        auto result_buffer = copy.to(vecmem::get_data(result), d_mem,
   vecmem::copy::type::host_to_device); auto result_view =
   vecmem::get_data(result_buffer);

      //  vecpar::cuda::parallel_map(second_alg, c,
                                 //  vecpar::cuda::parallel_map(first_alg, c,
   result_view, data_view));


        auto f1 = [&] <typename Algorithm, typename T, typename R = typename
   Algorithm::result_t, typename... Args>(Algorithm alg,
   vecmem::data::vector_view<T>& coll, Args... args) ->
   vecmem::data::vector_view<R>& { return vecpar::cuda::parallel_map(alg, c,
   result_view, coll, args...);
        };

        auto f2 = [&] <typename Algorithm, typename T, typename R = typename
   Algorithm::result_t, typename... Args>(Algorithm alg,
   vecmem::data::vector_view<T>& coll, Args... args) ->
   vecmem::data::vector_view<R>& { return vecpar::cuda::parallel_map(alg, c,
   coll, args...);
        };

        auto f3 = [&] <typename R>(vecmem::data::vector_view<R>& coll) -> R& {
            return vecpar::cuda::parallel_reduce(second_alg, c, coll);
        };

        double result_reduce = f3 (f2 (second_alg, f1 (first_alg, data_view)));
    //    copy(result_buffer1, result, vecmem::copy::type::device_to_host);

   //    vecpar::cuda::chain_orchestrator<int, vecmem::vector<double>> chain(mr,
   d_mem);
     //  chain.inputCollection(*vec).parallel_maps(f1, f2).retrieve_result();

       double sum = 0;
       for (int i = 0; i < vec->size(); i++) {
         //  printf("Compare %f with %f \n", result[i], vec->at(i) * 2.0);
   //        EXPECT_EQ(result[i], vec->at(i) * (2.0));
           sum += vec->at(i) * 2.0 ;
       }
       printf("result: %f\n", result_reduce);
        EXPECT_EQ(result_reduce, sum);


     //   chain().addConfig(c)
      //  .input(start_input)
       // .addStep(alg1, extra_args1)
        //.addStep(alg2, extra_args2)
      //  .addStep(alg3)

        //auto addStep = [&] <typename Algorithm, typename... Args>(Algorithm a,
   Args... args) {
          //  return vecpar::cuda::parallel_algorithm(a, class_var_c,
   start_view,)
       // };
    }
*/
/*
    TEST_P(ChainGpuTest, Parallel_Map_Map_Chained2) {

      //  test_algorithm_3 first_alg(mr);
       // test_algorithm_4 second_alg;

        test_algorithm_5 mmap;

        vecpar::config c{static_cast<int>(vec->size()/64 + 1), 64};

        vecpar::chain chain(mr, c);
        chain.add_algorithm<test_algorithm_5, double, X>(mmap);

    }
    */

TEST_P(ChainGpuTest, Parallel_Map_Map_Chained) {
  test_algorithm_3 first_alg(mr);
  test_algorithm_4 second_alg;

  vecpar::config c{static_cast<int>(vec->size() / 64 + 1), 64};

  int *d_data = NULL;
  cudaMalloc((void **)&d_data, vec->size() * sizeof(int));
  cudaMemcpy(d_data, vec->data(), vec->size() * sizeof(int),
             cudaMemcpyHostToDevice);
  vecpar::cuda_data<int> input{d_data, vec->size()};

  vecpar::cuda_data<double> d_result_2 = vecpar::cuda_raw::parallel_map(
      second_alg, c, vecpar::cuda_raw::parallel_map(first_alg, c, input));

  double *result = (double *)malloc(d_result_2.size * sizeof(double));
  cudaMemcpy(result, d_result_2.ptr, d_result_2.size * sizeof(double),
             cudaMemcpyDeviceToHost);

  for (int i = 0; i < d_result_2.size; i++)
    EXPECT_EQ(result[i], vec->at(i) * 2.0);
}

INSTANTIATE_TEST_SUITE_P(Chain_HostDevice, ChainGpuTest, testing::ValuesIn(N));
} // namespace
