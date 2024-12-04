/*! \file
 *
 * \brief Composite Struct
 *
 */

#pragma once
#include "all_struct.hh"

namespace holovibes
{

/*! \class ProgressInfo
 *
 * \brief TODO
 */
struct ProgressInfo
{
    ProgressType type;
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
    std::optional<uint> input_fps;
    std::optional<uint> output_fps;
    std::optional<uint> saving_fps;
    std::optional<uint> temperature;
    std::optional<ProgressInfo> file_read_progress;
    std::optional<ProgressInfo> record_progress;
    std::map<QueueType, QueueInfo> queues;
};

} // namespace holovibes