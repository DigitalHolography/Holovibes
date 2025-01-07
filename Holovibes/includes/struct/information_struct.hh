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
    std::shared_ptr<std::string> img_source;
    std::shared_ptr<std::string> input_format;
    std::shared_ptr<std::string> output_format;
    std::optional<ProgressInfo> file_read_progress;
    std::optional<ProgressInfo> record_progress;
    std::map<QueueType, QueueInfo> queues;
};

} // namespace holovibes