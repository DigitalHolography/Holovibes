/*!
 * \file fast_updates_types.hh
 *
 * \brief Defines the types used by the FastUpdatesHolder system
 *
 * Documentation on how to add entry types to the FastUpdatesHolder class
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
#pragma once

#include <atomic>
#include <cstddef>

#include "aliases.hh"
#include "enum_device.hh"

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

enum class IntType
{
    INPUT_FPS,
    OUTPUT_FPS,
    SAVING_FPS,
    TEMPERATURE,
};

enum class ProgressType
{
    FILE_READ,
    CHART_RECORD,
};

enum class RecordType
{
    FRAME,
};

// Queue identifiers from RECORD_QUEUE onward are reserved for record buffers.
enum class QueueType : std::size_t
{
    UNDEFINED,
    INPUT_QUEUE,
    OUTPUT_QUEUE,
    RECORD_QUEUE,
    RECORD_QUEUE_2,
    RECORD_QUEUE_3,
};

/*! \brief Get the queue identifier for a zero-based record buffer index. */
constexpr QueueType record_queue_type(std::size_t index)
{
    return static_cast<QueueType>(static_cast<std::size_t>(QueueType::RECORD_QUEUE) + index);
}

constexpr bool is_record_queue(QueueType type) { return type >= QueueType::RECORD_QUEUE; }

/*! \brief Get the zero-based buffer index of a record queue identifier. */
constexpr std::size_t record_queue_index(QueueType type)
{
    return static_cast<std::size_t>(type) - static_cast<std::size_t>(QueueType::RECORD_QUEUE);
}
} // namespace holovibes

/*! \} */

namespace holovibes::_internal
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
struct TypeValue<IntType>
{
    using key = IntType;
    using value = std::atomic<uint>;
};

template <>
struct TypeValue<ProgressType>
{
    using key = ProgressType;
    using value = std::pair<std::atomic<uint>, std::atomic<uint>>;
};

/*!
 * \brief entry type for record. 3 values : nb of frames acquired, nb of frames saved, total nb frames to acquire.
 */
template <>
struct TypeValue<RecordType>
{
    using key = RecordType;
    using value = std::tuple<std::atomic<uint>, std::atomic<uint>, std::atomic<uint>>;
};

/*!
 * \brief entry type for queues. 3 values : current occupancy of the queue, max size and device (GPU or CPU)
 */
template <>
struct TypeValue<QueueType>
{
    using key = QueueType;
    using value = std::tuple<std::atomic<uint>, std::atomic<uint>, std::atomic<holovibes::Device>>;
};

} // namespace holovibes::_internal
namespace holovibes
{

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
