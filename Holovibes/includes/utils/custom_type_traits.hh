#pragma once

/**
 * @file custome_type_traits.hh
 * This file provides a variety of type traits.
 */

#include <tuple>
#include <type_traits>

/**
 * @brief SFINAE helper to enable a function if the type is any of the given
 * types.
 * @tparam T The type to check.
 * @tparam TYPES The types to check against.
 */
template <typename T, typename... TYPES>
struct is_any_of : std::false_type
{
};

/**
 * @brief SFINEA helper to enable a function if the type is any of the given
 * types.
 * @tparam T The type to check.
 * @tparam FIRST The first type to check against.
 * @tparam ...TYPES The remaining types to check against.
 */
template <typename T, typename FIRST, typename... TYPES>
struct is_any_of<T, FIRST, TYPES...>
    : std::bool_constant<std::is_same<T, FIRST>::value || is_any_of<T, TYPES...>::value>
{
};

/**
 * @brief SFINEA helper to enable a function if the type is any of the given
 * types.
 * @tparam T The type to check.
 * @tparam ...TYPES The types to check against.
 */
template <typename T, typename... TYPES>
using enable_if_any_of = std::enable_if_t<is_any_of<T, TYPES...>::value>;

/**
 * @brief Concept to check if a type is a tuple-like type.
 * @tparam T the type to check.
 */
template <typename T>
concept TupleLike = requires(T t) {
    std::tuple_size<T>::value;
    std::get<0>(t);
};

/**
 * @brief SFINAE helper to check if a tuple contains a given Type.
 * @tparam T to type to check.
 * @tparam Tuple the tuple that should contains the type.
 */
template <typename T, typename Tuple>
struct tuple_has_type : std::false_type
{
};

/**
 * @brief SFINAE helper to check if a tuple contains a given type.
 * @tparam T to type to check.
 * @tparam Types the types contained in the tuple.
 */
template <typename T, typename... Types>
struct tuple_has_type<T, std::tuple<Types...>> : is_any_of<T, Types...>
{
};

/**
 * @brief Concept to check if a tuple contains at least all the given types.
 * @tparam T The tuple;
 * @tparam Types The types that should be in the tuple.
 */
template <typename T, typename... Types>
concept TupleContainsTypes = requires(T t) {
    // Check T is a tuple.
    std::tuple_size<T>::value;

    // Check T contains all the given types.
    requires(tuple_has_type<Types, T>::value && ...);
};
