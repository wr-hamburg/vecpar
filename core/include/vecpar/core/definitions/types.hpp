#ifndef VECPAR_TYPES_HPP
#define VECPAR_TYPES_HPP

#include <vecmem/containers/jagged_vector.hpp>
#include <vecmem/containers/vector.hpp>

namespace vecpar::collection {

/// check if T is a vecmem::vector
template <typename T>
concept Vector_type = std::same_as<T, vecmem::vector<typename T::value_type>>;

/// check if T is a vecmem::jagged_vectpr
template <typename T>
concept Jagged_vector_type =
    std::same_as<T, vecmem::jagged_vector<typename T::value_type::value_type>>;

/// check if T is vector or jagged_vector
template <typename T>
concept Iterable = Vector_type<T> || Jagged_vector_type<T>;

/// https://stackoverflow.com/questions/62203496/type-trait-to-receive-tvalue-type-if-present-t-otherwise
template <class T, class = void> struct value_type { using type = T; };

template <class T> struct value_type<T, std::void_t<typename T::value_type>> {
  using type = typename T::value_type;
};

template <Jagged_vector_type T>
struct value_type<T, std::void_t<typename T::value_type>> {
  using type = typename T::value_type::value_type;
};

template <class T> using value_type_t = typename value_type<T>::type;

// https://stackoverflow.com/questions/11056714/c-type-traits-to-extract-template-parameter-class
template <typename T> struct extract_value_type { typedef T value_type; };

template <template <typename> class X, typename T>
struct extract_value_type<X<T>> // specialization
{
  typedef T value_type;
};

template <typename> struct is_iterable : std::false_type {};

template <typename T, typename A>
struct is_iterable<std::vector<T, A>> : std::true_type {};

/// check if T is vector_view or jagged_vector_view
template <typename T>
concept jagged_view =
    std::is_same_v<T, vecmem::data::jagged_vector_data<
                          typename extract_value_type<T>::value_type>>;

/// retrieve the view from a vector/jagged_vector or object unmodified otherwise
template <typename... T>
std::tuple<std::conditional_t<
    (std::is_object<T>::value && Jagged_vector_type<T>),
    vecmem::data::jagged_vector_data<value_type_t<T>>,
    std::conditional_t<(std::is_object<T>::value && Vector_type<T>),
                       vecmem::data::vector_view<value_type_t<T>>, T>>...>
get_view_or_obj(T &...obj) {
  return {([](T &i) {
    if constexpr (Jagged_vector_type<T>) {
      auto data = vecmem::get_data(i);
      return data;
    } else if constexpr (Vector_type<T>) {
      auto view = vecmem::get_data(i);
      return view;
    } else {
      return i;
    }
  }(obj))...};
}

/**
 * The number of iterable collections in the algorithm's input
 */
enum count { One, Two, Three, Four, Five };

} // namespace vecpar::collection
#endif // VECPAR_TYPES_HPP
