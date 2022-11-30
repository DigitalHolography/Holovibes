#include "API.hh"
namespace holovibes
{

template <>
void ExportPipeRequestOnSync::operator()<FrameRecord>(const FrameRecordStruct& new_value, Pipe& pipe)
{
    LOG_UPDATE_ON_SYNC(FrameRecord);

    if (new_value.is_running == false)
    {
        pipe.get_frame_record_env().gpu_frame_record_queue_.reset(nullptr);
        Holovibes::instance().stop_frame_record();
        return;
    }

    if (new_value.record_type == FrameRecordStruct::RecordType::HOLOGRAM)
    {
        auto record_fd = api::get_output_frame_descriptor();
        record_fd.depth = record_fd.depth == 6 ? 3 : record_fd.depth;
        pipe.get_frame_record_env().gpu_frame_record_queue_.reset(
            new Queue(record_fd, api::detail::get_value<RecordBufferSize>(), QueueType::RECORD_QUEUE));
    }
    else if (new_value.record_type == FrameRecordStruct::RecordType::RAW)
    {
        pipe.get_frame_record_env().gpu_frame_record_queue_.reset(new Queue(api::get_import_frame_descriptor(),
                                                                            api::detail::get_value<RecordBufferSize>(),
                                                                            QueueType::RECORD_QUEUE));
    }
    else if (new_value.record_type == FrameRecordStruct::RecordType::CUTS_XZ ||
             new_value.record_type == FrameRecordStruct::RecordType::CUTS_YZ)
    {
        FrameDescriptor fd_xyz = api::get_output_frame_descriptor();

        fd_xyz.depth = sizeof(ushort);
        if (new_value.record_type == FrameRecordStruct::RecordType::CUTS_XZ)
            fd_xyz.height = api::detail::get_value<TimeTransformationSize>();
        else if (new_value.record_type == FrameRecordStruct::RecordType::CUTS_YZ)
            fd_xyz.width = api::detail::get_value<TimeTransformationSize>();

        pipe.get_frame_record_env().gpu_frame_record_queue_.reset(
            new Queue(fd_xyz, api::detail::get_value<RecordBufferSize>(), QueueType::RECORD_QUEUE));
    }

    Holovibes::instance().start_frame_record();

    request_pipe_refresh();
}

template <>
void ExportPipeRequestOnSync::operator()<ChartRecord>(const ChartRecordStruct& new_value, Pipe& pipe)
{
    LOG_UPDATE_ON_SYNC(ChartRecord);

    if (new_value.is_running == false)
    {
        pipe.get_chart_env().chart_record_queue_.reset(nullptr);
        Holovibes::instance().stop_chart_record();
    }
    else
    {
        pipe.get_chart_env().chart_record_queue_.reset(new ConcurrentDeque<ChartPoint>());
        Holovibes::instance().start_chart_record();
    }
}
} // namespace holovibes
