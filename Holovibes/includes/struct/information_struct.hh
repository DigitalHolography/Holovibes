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
 * \brief Small structure for holding a value / max value pair used for progress data
 */
struct ProgressInfo
{
    uint current_size;
    uint max_size;
};

/*! \class GpuInfo
 *
 * \brief Small structure that holds the GPU and GPU memory controller loads
 */
struct GpuInfo
{
    uint gpu;
    uint memory;
    size_t controller_memory;
    size_t controller_total;
};

/*! \class QueueInfo
 *
 * \brief Small structure that holds info regarding the state of a queue
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
    size_t input_throughput;
    size_t output_throughput;
    size_t saving_throughput;
    std::shared_ptr<std::string> img_source;
    std::shared_ptr<std::string> input_format;
    std::shared_ptr<std::string> output_format;
    std::optional<GpuInfo> gpu_info;
    std::map<ProgressType, ProgressInfo> progresses;
    std::map<QueueType, QueueInfo> queues;
};

} // namespace holovibes