#include <vecmem/containers/vector.hpp>
#include <vecmem/containers/data/vector_view.hpp>
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/cuda/managed_memory_resource.hpp>
#include <vecmem/utils/cuda/copy.hpp>
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/memory/host_memory_resource.hpp>
#include <iostream>
#include <cuda.h>

 vecmem::cuda::device_memory_resource d_mem;
 vecmem::cuda::copy copy;

template <typename Function, typename... Arguments>
__global__ void kernel(size_t size, Function f, Arguments... args) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;
    f(idx, args...);
}

template<typename T>
concept is_vec = std::is_base_of<vecmem::vector<typename T::value_type>, T>::value;

template<class T, class = void>
struct value_type {
    using type = T;
};

template<class T>
struct value_type<T, std::void_t<typename T::value_type>> {
    using type = typename T::value_type;
};

template<class T> using value_type_t = typename value_type<T>::type;

template<typename... T>
std::tuple<std::conditional_t<(std::is_object<T>::value && is_vec<T>),
        vecmem::data::vector_view<value_type_t<T>>, T>...>
get_view_of_copied_container_or_obj(T &... obj) {
    return {([](T &i) {
        if constexpr (is_vec<T>) {
            auto buffer = copy.to(vecmem::get_data(i), d_mem,
                                  vecmem::copy::type::host_to_device);
            auto view = vecmem::get_data(buffer);
            return view;
        } else {
            return i;
        }
    }(obj))...};

}

template <typename T1, typename... Args>
void g(vecmem::data::vector_view<value_type_t<T1>>& vec,
       Args... args ) {

    std::cout << "calling g with T1" << std::endl;

    kernel<<<1,2>>>(2,
                    [=] __device__(int idx, Args... a) mutable {
                        vecmem::device_vector<value_type_t<T1>> d_vec(vec);
                        printf("Reaching here\n");
                        printf("vec[%d]=%d\n", idx, d_vec[idx]);
                        printf("a=%d\n",a...);
                    },
    args...);
    cudaDeviceSynchronize();
}

template <typename T1, typename T2, typename... Args>
void g(vecmem::data::vector_view<value_type_t<T1>>& vec1,
       vecmem::data::vector_view<value_type_t<T2>>& vec2,
       Args... args ) {
    std::cout << "calling g with T1 and T2" << std::endl;

    kernel<<<1,2>>>(vec1.size(),
            [=] __device__(int idx, Args... a) mutable {
        vecmem::device_vector<value_type_t<T1>> d_vec1(vec1);
        vecmem::device_vector<value_type_t<T2>> d_vec2(vec2);
        printf("vec1[%d]=%d and vec2[%d]=%f\n", idx, d_vec1[idx], idx, d_vec2[idx]);
        printf("a=%d\n",a...);
    },
    args...);
    cudaDeviceSynchronize();
}

template <typename... Arguments>
void f(Arguments&... args) {
    auto params = get_view_of_copied_container_or_obj(args...);

    auto fn = [&] <typename... T> (T&... types) {
        return g<Arguments...>(types...);
    };

    std::apply(fn, params);
}

int main() {

    int value = -1;


    vecmem::host_memory_resource mem;
  //  vecmem::cuda::managed_memory_resource mem;

    vecmem::vector<int> v1(2, &mem);
    v1[0]=1;
    v1[1]=2;

    vecmem::vector<float> v2(2, &mem);
    v2[0]=9.0;
    v2[1]=10.0;

    f(v1, value);
    f<vecmem::vector<int>, vecmem::vector<float>, int>(v1, v2, value);
}

/*
add_executable(test "scenarios/Test.cpp")
#set_target_properties(test PROPERTIES CXX_STANDARD 20)
target_compile_options(test PUBLIC
        $<$<COMPILE_LANGUAGE:CXX>:-std=c++20 -Wall -Wextra -Wpedantic>
$<$<COMPILE_LANGUAGE:CXX>:-x cuda --offload-arch=sm_86>
                                                 $<$<COMPILE_LANGUAGE:CXX>:-DVECMEM_HAVE_PMR_MEMORY_RESOURCE>)
target_link_libraries(test vecmem::core vecmem::cuda CUDA::cudart)
 */