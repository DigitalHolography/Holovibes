#pragma once

#include <string>
#include <atomic>
#include <type_traits>

#include "queue.hh"

/*!
 * \brief Documentation on how to add entry types to the FastUpdatesHolder class
 *
 * First decide for an Enum name (named TKey in the following) and an Value type (named TValue in the following)
 *
 * Then you new to create (or import as done with QueueType) an enum with the entries you want
 * and finaly you need to create a specialization of the templated struct TypeValue as follows:
 *
 * template <>
 * struct TypeValue<TKey>
 * {
 *     using key = TKey;
 *     using value = TValue;
 * };
 */

namespace holovibes
{

/*! \name Key types of the FastUpdatesHolder class
 * \{
 */

enum class IndicationType
{
    IMG_SOURCE,
    INPUT_FORMAT,
    OUTPUT_FORMAT,
};

enum class FpsType
{
    INPUT_FPS,
    OUTPUT_FPS,
    SAVING_FPS,
};

enum class ProgressType
{
    FILE_READ,
    FRAME_RECORD,
    CHART_RECORD,
};

// enum class also
using QueueType = Queue::QueueType;

/*! \} */

namespace _internal
{

template <typename T>
struct TypeValue
{
    using key = T;
    using value = std::false_type;
};

template <>
struct TypeValue<IndicationType>
{
    using key = IndicationType;
    using value = std::string;
};

template <>
struct TypeValue<FpsType>
{
    using key = FpsType;
    using value = std::atomic<unsigned int>;
};

template <>
struct TypeValue<ProgressType>
{
    using key = ProgressType;
    using value = std::atomic<unsigned int>;
};

} // namespace _internal

template <typename T>
constexpr bool is_fast_update_key_type = !std::is_same<_internal::TypeValue<T>::value, std::false_type>::value;

template <typename T>
using FastUpdateTypeValue = _internal::TypeValue<T>::value;

} // namespace holovibes
