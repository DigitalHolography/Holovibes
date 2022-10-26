#pragma once

#include "API.hh"

namespace holovibes
{

template <>
void ExportPipeRequestOnSync::operator()<FrameRecord>(const FrameRecordStruct& new_value,
                                                      const FrameRecordStruct& old_value,
                                                      Pipe& pipe)
{
    LOG_TRACE(compute_worker, "UPDATE FrameRecord");

    if (new_value.get_record_mode() == RecordMode::HOLOGRAM)
    {
        auto record_fd = gpu_output_queue_.get_fd();
        record_fd.depth = record_fd.depth == 6 ? 3 : record_fd.depth;
        frame_record_env_.gpu_frame_record_queue_.reset(
            new Queue(record_fd, advanced_cache_.get_value<RecordBufferSize>(), QueueType::RECORD_QUEUE));
    }
    else if (new_value.get_record_mode() == RecordMode::RAW)
    {
        frame_record_env_.gpu_frame_record_queue_.reset(new Queue(gpu_input_queue_.get_fd(),
                                                                  advanced_cache_.get_value<RecordBufferSize>(),
                                                                  QueueType::RECORD_QUEUE));
    }
    else if (new_value.get_record_mode() == RecordMode::CUTS_XZ || new_value.get_record_mode() == RecordMode::CUTS_YZ)
    {
        camera::FrameDescriptor fd_xyz = gpu_output_queue_.get_fd();

        fd_xyz.depth = sizeof(ushort);
        if (new_value.get_record_mode() == RecordMode::CUTS_XZ)
            fd_xyz.height = GSH::instance().get_value<TimeTransformationSize>();
        else if (new_value.get_record_mode() == RecordMode::CUTS_YZ)
            fd_xyz.width = GSH::instance().get_value<TimeTransformationSize>();

        frame_record_env_.gpu_frame_record_queue_.reset(
            new Queue(fd_xyz, GSH::instance().get_value<RecordBufferSize>(), QueueType::RECORD_QUEUE));
    }
}

template <>
void ExportPipeRequestOnSync::operator()<ChartRecord>(const ChartRecordStruct& new_value,
                                                      const ChartRecordStruct& old_value,
                                                      Pipe& pipe)
{
    LOG_TRACE(compute_worker, "UPDATE ChartRecord");

    if (new_value.get_is_enabled() == false)
    {
        if (new_value.get_is_enabled() == old_value.get_is_enabled())
            return;
        pipe.get_chart_env().chart_record_queue_.reset(nullptr);
    }

    if (new_value.get_nb_points_to_record() == old_value.get_nb_points_to_record())
        return;

    pipe.get_chart_env().chart_record_queue_.reset(new ConcurrentDeque<ChartPoint>());
}
} // namespace holovibes
