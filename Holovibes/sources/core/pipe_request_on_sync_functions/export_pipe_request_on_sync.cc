#include "API.hh"
namespace holovibes
{

template <>
void ExportPipeRequestOnSync::operator()<Record>(const RecordStruct& new_value, Pipe& pipe)
{
    LOG_UPDATE_ON_SYNC(Record);

    if (new_value.is_running == false)
    {
        if (new_value.record_type == RecordStruct::RecordType::CHART)
        {
            Holovibes::instance().stop_and_join_chart_record();
            pipe.get_chart_env().chart_record_queue_.reset(nullptr);
        }
        else
        {
            Holovibes::instance().stop_and_join_frame_record();
            pipe.get_frame_record_env().gpu_frame_record_queue_.reset(nullptr);
        }
        request_pipe_refresh();
        return;
    }

    // CHART MODE
    if (new_value.record_type == RecordStruct::RecordType::CHART)
    {
        pipe.get_chart_env().chart_record_queue_.reset(new ConcurrentDeque<ChartPoint>());
        Holovibes::instance().stop_and_join_chart_record();
    }
    // FRAME MODE
    else if (new_value.record_type == RecordStruct::RecordType::HOLOGRAM)
    {
        auto record_fd = api::get_output_frame_descriptor();
        record_fd.depth = record_fd.depth == 6 ? 3 : record_fd.depth;
        pipe.get_frame_record_env().gpu_frame_record_queue_.reset(
            new Queue(record_fd, api::detail::get_value<RecordBufferSize>(), QueueType::RECORD_QUEUE));
        Holovibes::instance().stop_and_join_frame_record();
    }
    else if (new_value.record_type == RecordStruct::RecordType::RAW)
    {
        pipe.get_frame_record_env().gpu_frame_record_queue_.reset(new Queue(api::get_import_frame_descriptor(),
                                                                            api::detail::get_value<RecordBufferSize>(),
                                                                            QueueType::RECORD_QUEUE));
        Holovibes::instance().stop_and_join_frame_record();
    }
    else if (new_value.record_type == RecordStruct::RecordType::CUTS_XZ ||
             new_value.record_type == RecordStruct::RecordType::CUTS_YZ)
    {
        FrameDescriptor fd_xyz = api::get_output_frame_descriptor();

        fd_xyz.depth = sizeof(ushort);
        if (new_value.record_type == RecordStruct::RecordType::CUTS_XZ)
            fd_xyz.height = api::detail::get_value<TimeTransformationSize>();
        else if (new_value.record_type == RecordStruct::RecordType::CUTS_YZ)
            fd_xyz.width = api::detail::get_value<TimeTransformationSize>();

        pipe.get_frame_record_env().gpu_frame_record_queue_.reset(
            new Queue(fd_xyz, api::detail::get_value<RecordBufferSize>(), QueueType::RECORD_QUEUE));
        Holovibes::instance().stop_and_join_frame_record();
    }

    request_pipe_refresh();
}

} // namespace holovibes
