#ifndef VECPAR_TYPES_HPP
#define VECPAR_TYPES_HPP

#include <vecmem/containers/jagged_vector.hpp>
#include <vecmem/containers/vector.hpp>

namespace vecpar::collection {

template <typename T>
concept Vector_type = std::same_as<T, vecmem::vector<typename T::value_type>>;

template <typename T>
concept Jagged_vector_type =
    std::same_as<T, vecmem::jagged_vector<typename T::value_type>>;

template <typename T>
concept Iterable = Vector_type<T> || Jagged_vector_type<T>;

/**
 * The number of iterable collections in the algorithm's input
 */
enum count { One, Two, Three, Four, Five };
} // namespace vecpar::collection
#endif // VECPAR_TYPES_HPP
