#include "API.hh"

namespace holovibes
{

template <>
void ExportPipeRequestOnSync::operator()<FrameRecordMode>(const FrameRecordStruct& new_value,
                                                          const FrameRecordStruct& old_value,
                                                          Pipe& pipe)
{
    LOG_TRACE(compute_worker, "UPDATE FrameRecord");

    if (new_value.get_record_mode() == RecordMode::NONE)
    {
        pipe.get_frame_record_env().gpu_frame_record_queue_.reset(nullptr);
    }
    else if (new_value.get_record_mode() == RecordMode::HOLOGRAM)
    {
        auto record_fd = pipe.get_gpu_output_queue().get_fd();
        record_fd.depth = record_fd.depth == 6 ? 3 : record_fd.depth;
        pipe.get_frame_record_env().gpu_frame_record_queue_.reset(
            new Queue(record_fd, pipe.get_advanced_cache().get_value<RecordBufferSize>(), QueueType::RECORD_QUEUE));
    }
    else if (new_value.get_record_mode() == RecordMode::RAW)
    {
        pipe.get_frame_record_env().gpu_frame_record_queue_.reset(
            new Queue(api::get_gpu_input_queue().get_fd(),
                      pipe.get_advanced_cache().get_value<RecordBufferSize>(),
                      QueueType::RECORD_QUEUE));
    }
    else if (new_value.get_record_mode() == RecordMode::CUTS_XZ || new_value.get_record_mode() == RecordMode::CUTS_YZ)
    {
        camera::FrameDescriptor fd_xyz = pipe.get_gpu_output_queue().get_fd();

        fd_xyz.depth = sizeof(ushort);
        if (new_value.get_record_mode() == RecordMode::CUTS_XZ)
            fd_xyz.height = GSH::instance().get_value<TimeTransformationSize>();
        else if (new_value.get_record_mode() == RecordMode::CUTS_YZ)
            fd_xyz.width = GSH::instance().get_value<TimeTransformationSize>();

        pipe.get_frame_record_env().gpu_frame_record_queue_.reset(
            new Queue(fd_xyz, GSH::instance().get_value<RecordBufferSize>(), QueueType::RECORD_QUEUE));
    }
}

template <>
void ExportPipeRequestOnSync::operator()<ChartRecord>(const ChartRecordStruct& new_value,
                                                      const ChartRecordStruct& old_value,
                                                      Pipe& pipe)
{
    LOG_TRACE(compute_worker, "UPDATE ChartRecord");

    if (new_value.is_enable() == false)
        pipe.get_chart_env().chart_record_queue_.reset(nullptr);
    else
        pipe.get_chart_env().chart_record_queue_.reset(new ConcurrentDeque<ChartPoint>());
}
} // namespace holovibes
