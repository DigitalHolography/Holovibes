#include "frame_record_worker.hh"
#include "output_frame_file_factory.hh"
#include "tools.hh"
#include "holovibes.hh"
#include "icompute.hh"
#include "global_state_holder.hh"
#include "API.hh"
#include "logger.hh"

namespace holovibes::worker
{
FrameRecordWorker::FrameRecordWorker()
    : Worker()
    , env_(api::get_compute_pipe().get_frame_record_env())
    , stream_(Holovibes::instance().get_cuda_streams().recorder_stream)
{
    auto& entry = GSH::fast_updates_map<ProgressType>.create_entry(ProgressType::FRAME_RECORD);
    entry.recorded = &env_.current_nb_frames_recorded;
    entry.to_record = &export_cache_.get_value<FrameRecord>().nb_frames_to_record;

    GSH::fast_updates_map<FpsType>.create_entry(FpsType::SAVING_FPS) = &processed_fps_;
}

void FrameRecordWorker::integrate_fps_average()
{
    // An fps of 0 is not relevent. We do not includx it in fps average.
    if (current_fps_ == 0)
        return;

    fps_buffer_[fps_current_index_++ % FPS_LAST_X_VALUES] = current_fps_;
}

size_t FrameRecordWorker::compute_fps_average() const
{
    LOG_FUNC();

    if (fps_current_index_ == 0)
        return 0;

    size_t ret = 0;
    size_t upper = FPS_LAST_X_VALUES < fps_current_index_ ? FPS_LAST_X_VALUES : fps_current_index_;
    for (size_t i = 0; i < upper; i++)
        ret += fps_buffer_[i];

    ret /= upper;

    return ret;
}

void FrameRecordWorker::run()
{
    LOG_FUNC();
    // Progress recording FastUpdatesHolder entry

    // Init vars
    const auto& nb_frames_to_record = export_cache_.get_value<FrameRecord>().nb_frames_to_record;
    env_.current_nb_frames_recorded = 0;
    env_.nb_frame_skip = export_cache_.get_value<FrameRecord>().nb_frames_to_skip;
    processed_fps_ = 0;
    
    const size_t output_frame_size = env_.gpu_frame_record_queue_->get_fd().get_frame_size();
    io_files::OutputFrameFile* output_frame_file = nullptr;
    char* frame_buffer = nullptr;

    try
    {
        output_frame_file =
            io_files::OutputFrameFileFactory::create(export_cache_.get_value<FrameRecord>().frames_file_path,
                                                     env_.gpu_frame_record_queue_->get_fd(),
                                                     nb_frames_to_record);

        output_frame_file->write_header();

        std::optional<int> contiguous_frames = std::nullopt;

        frame_buffer = new char[output_frame_size];

        while (nb_frames_to_record == 0 || env_.current_nb_frames_recorded < nb_frames_to_record)
        {
            if (stop_requested_)
                break;

            if (env_.gpu_frame_record_queue_->has_overridden() ||
                Holovibes::instance().get_gpu_input_queue()->has_overridden())
            {
                // Due to frames being overwritten when the queue/batchInputQueue is full, the contiguity is lost.
                if (!contiguous_frames.has_value())
                    contiguous_frames = std::make_optional(env_.current_nb_frames_recorded);
            }

            wait_for_frames(*env_.gpu_frame_record_queue_);

            // While wait_for_frames() is running, a stop might be requested and the queue reset.
            // To avoid problems with dequeuing while it's empty, we check right after wait_for_frame
            // and stop recording if needed.
            if (stop_requested_)
                break;

            if (env_.nb_frame_skip > 0)
            {
                env_.gpu_frame_record_queue_->dequeue();
                env_.nb_frame_skip--;
                continue;
            }

            env_.gpu_frame_record_queue_->dequeue(frame_buffer, stream_, cudaMemcpyDeviceToHost);
            output_frame_file->write_frame(frame_buffer, output_frame_size);
            processed_fps_++;
            env_.current_nb_frames_recorded++;

            integrate_fps_average();
        }

        LOG_INFO("Recording stopped, written frames : {}", env_.current_nb_frames_recorded);
        output_frame_file->correct_number_of_frames(env_.current_nb_frames_recorded);

        if (contiguous_frames.has_value())
        {
            LOG_WARN("Record lost its contiguousity at frame {}.", contiguous_frames.value());
            LOG_WARN("To prevent this lost, you might need to increase Input AND/OR Record buffer size.");
        }
        else
        {
            LOG_INFO("Record is contiguous!");
        }

        auto contiguous = contiguous_frames.value_or(env_.current_nb_frames_recorded);
        output_frame_file->export_compute_settings(compute_fps_average(), contiguous);

        output_frame_file->write_footer();
    }
    catch (const io_files::FileException& e)
    {
        LOG_INFO("{}", e.what());
    }

    delete output_frame_file;
    delete[] frame_buffer;

    reset_gpu_record_queue();

    GSH::fast_updates_map<ProgressType>.remove_entry(ProgressType::FRAME_RECORD);
    GSH::fast_updates_map<FpsType>.remove_entry(FpsType::SAVING_FPS);

    // LOG_TRACE(record_worker, "Exiting FrameRecordWorker::run()");
}

void FrameRecordWorker::wait_for_frames(Queue& record_queue)
{
    while (!stop_requested_)
    {
        if (env_.gpu_frame_record_queue_->get_size() != 0)
            break;
    }
}

void FrameRecordWorker::reset_gpu_record_queue()
{
    api::detail::change_value<FrameRecord>()->is_running = false;

    std::unique_ptr<Queue>& raw_view_queue = api::get_compute_pipe().get_raw_view_queue_ptr();
    if (raw_view_queue)
        raw_view_queue->resize(api::get_output_buffer_size(), stream_);

    std::shared_ptr<Queue> output_queue = Holovibes::instance().get_gpu_output_queue();
    if (output_queue)
        output_queue->resize(api::get_output_buffer_size(), stream_);
}
} // namespace holovibes::worker
