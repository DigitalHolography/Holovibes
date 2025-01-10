/*! \file
 *
 * \brief Composite Struct
 *
 */

#pragma once

#include <atomic>
#include <map>
#include <memory>

#include "all_struct.hh"
#include "enum_device.hh"
#include "fast_updates_types.hh"

typedef unsigned int uint;

namespace holovibes
{

/*! \class ProgressInfo
 *
 * \brief TODO
 */
struct ProgressInfo
{
    uint current_size;
    uint max_size;
};

/*! \class QueueInfo
 *
 * \brief TODO
 */
struct QueueInfo
{
    uint current_size;
    uint max_size;
    Device device;
};

/*! \class Information
 *
 * \brief TODO
 */
struct Information
{
    std::optional<size_t> input_fps;
    std::optional<size_t> output_fps;
    std::optional<size_t> saving_fps;
    std::optional<size_t> temperature;
    std::shared_ptr<std::string> img_source;
    std::shared_ptr<std::string> input_format;
    std::shared_ptr<std::string> output_format;
    std::map<ProgressType, ProgressInfo> progresses;
    std::map<QueueType, QueueInfo> queues;
};

} // namespace holovibes