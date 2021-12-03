#pragma once

/*!
 * \brief Documentation on how to add entry types to the FastUpdatesHolder class
 *
 * First decide for an Enum name (named TKey in the following) and an Value type (named TValue in the following)
 *
 * Then you need to create (or import as done with QueueType) an enum with the entries you want
 * and finally you need to create a specialization of the templated struct TypeValue as follows:
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

enum class QueueType
{
    UNDEFINED,
    INPUT_QUEUE,
    OUTPUT_QUEUE,
    RECORD_QUEUE,
};

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
    using value = std::atomic<uint>;
};

template <>
struct TypeValue<ProgressType>
{
    using key = ProgressType;
    using value = std::pair<std::atomic<uint>, std::atomic<uint>>;
};

template <>
struct TypeValue<QueueType>
{
    using key = QueueType;
    using value = std::pair<std::atomic<uint>, std::atomic<uint>>;
};

} // namespace _internal

/*!
 * \brief Compile time
 *
 * \tparam T The enum class type to get the value of the key
 *  please do not use before checking is_fast_update_key_type
 */
template <typename T>
using FastUpdateTypeValue = typename _internal::TypeValue<T>::value;

/*!
 * \brief compile time boolean to check if the type T matches a key type of the FastUpdateHolder map class
 *
 * \tparam T The enum class type to check
 */
template <typename T>
static constexpr bool is_fast_update_key_type = !std::is_same<FastUpdateTypeValue<T>, std::false_type>::value;

} // namespace holovibes
