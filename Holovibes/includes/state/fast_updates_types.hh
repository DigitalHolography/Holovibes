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

#include "types.hh"
#include "json_macro.hh"
#include "logger.hh"

namespace holovibes
{

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
    READ,
    RECORD,
};

enum class QueueType
{
    UNDEFINED,
    INPUT_QUEUE,
    OUTPUT_QUEUE,
    RECORD_QUEUE,
};

// clang-format off
SERIALIZE_JSON_ENUM(IndicationType, {
    {IndicationType::IMG_SOURCE, "IMG_SOURCE"},
    {IndicationType::INPUT_FORMAT, "INPUT_FORMAT"},
    {IndicationType::OUTPUT_FORMAT, "OUTPUT_FORMAT"},
})
// clang-format on

inline std::ostream& operator<<(std::ostream& os, const IndicationType& value) { return os << json{value}; }

// clang-format off
SERIALIZE_JSON_ENUM(FpsType, {
    {FpsType::INPUT_FPS, "INPUT_FPS"},
    {FpsType::OUTPUT_FPS, "OUTPUT_FPS"},
    {FpsType::SAVING_FPS, "SAVING_FPS"},
})
// clang-format on

inline std::ostream& operator<<(std::ostream& os, const FpsType& value) { return os << json{value}; }

// clang-format off
SERIALIZE_JSON_ENUM(ProgressType, {
    {ProgressType::READ, "READ"},
    {ProgressType::RECORD, "RECORD"},
})
// clang-format on

inline std::ostream& operator<<(std::ostream& os, const ProgressType& value) { return os << json{value}; }

// clang-format off
SERIALIZE_JSON_ENUM(QueueType, {
    {QueueType::UNDEFINED, "UNDEFINED"},
    {QueueType::INPUT_QUEUE, "INPUT_QUEUE"},
    {QueueType::OUTPUT_QUEUE, "OUTPUT_QUEUE"},
    {QueueType::RECORD_QUEUE, "RECORD_QUEUE"},
})
// clang-format on

inline std::ostream& operator<<(std::ostream& os, const QueueType& value) { return os << json{value}; }
} // namespace holovibes

namespace holovibes::internal
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
    struct value
    {
        uint image_processed = 0;
    };
};

template <>
struct TypeValue<ProgressType>
{
    using key = ProgressType;
    struct value
    {
        uint* recorded = nullptr;
        const uint* to_record = nullptr;
    };
};

template <>
struct TypeValue<QueueType>
{
    using key = QueueType;
    struct value
    {
        const std::atomic<uint>* size = nullptr;
        const uint* max_size = nullptr;
    };
};

} // namespace holovibes::internal
namespace holovibes
{

/*!
 * \brief Compile time
 *
 * \tparam T The enum class type to get the value of the key
 *  please do not use before checking is_fast_update_key_type
 */
template <typename T>
using FastUpdateTypeValue = typename internal::TypeValue<T>::value;

/*!
 * \brief compile time boolean to check if the type T matches a key type of the FastUpdateHolder map class
 *
 * \tparam T The enum class type to check
 */
template <typename T>
static constexpr bool is_fast_update_key_type = !std::is_same<FastUpdateTypeValue<T>, std::false_type>::value;

} // namespace holovibes
