#include "frame_record_worker.hh"
#include "output_frame_file_factory.hh"
#include "tools.hh"
#include "holovibes.hh"
#include "icompute.hh"
#include "global_state_holder.hh"
#include "API.hh"
#include "logger.hh"
#include <spdlog/spdlog.h>

namespace holovibes::worker
{
FrameRecordWorker::FrameRecordWorker(const std::string& file_path,
                                     std::optional<unsigned int> nb_frames_to_record,
                                     RecordMode record_mode,
                                     unsigned int nb_frames_skip,
                                     const unsigned int output_buffer_size)
    : Worker()
    , file_path_(get_record_filename(file_path))
    , nb_frames_to_record_(nb_frames_to_record)
    , nb_frames_skip_(nb_frames_skip)
    , record_mode_(record_mode)
    , output_buffer_size_(output_buffer_size)
    , stream_(Holovibes::instance().get_cuda_streams().recorder_stream)
{
}

void FrameRecordWorker::integrate_fps_average()
{
    auto& fps_map = GSH::fast_updates_map<FpsType>;
    auto input_fps = fps_map.get_entry(FpsType::INPUT_FPS);
    int current_fps = input_fps->load();

    // An fps of 0 is not relevent. We do not includ it in fps average.
    if (current_fps == 0)
        return;

    fps_buffer_[fps_current_index_++ % FPS_LAST_X_VALUES] = current_fps;
}

size_t FrameRecordWorker::compute_fps_average() const
{
    LOG_FUNC();
    spdlog::trace("fps_current_index_ = {}", fps_current_index_);


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

    auto fast_update_progress_entry = GSH::fast_updates_map<ProgressType>.create_entry(ProgressType::FRAME_RECORD);
    std::atomic<uint>& nb_frames_recorded = fast_update_progress_entry->first;
    std::atomic<uint>& nb_frames_to_record = fast_update_progress_entry->second;

    nb_frames_recorded = 0;

    if (nb_frames_to_record_.has_value())
        nb_frames_to_record = nb_frames_to_record_.value();
    else
        nb_frames_to_record = 0;

    // Processed FPS FastUpdatesHolder entry

    std::shared_ptr<std::atomic<uint>> processed_fps = GSH::fast_updates_map<FpsType>.create_entry(FpsType::SAVING_FPS);
    *processed_fps = 0;
    Queue& record_queue = init_gpu_record_queue();
    const size_t output_frame_size = record_queue.get_fd().get_frame_size();
    io_files::OutputFrameFile* output_frame_file = nullptr;
    char* frame_buffer = nullptr;

    try
    {
        output_frame_file =
            io_files::OutputFrameFileFactory::create(file_path_, record_queue.get_fd(), nb_frames_to_record);
        LOG_DEBUG("output_frame_file = {}", output_frame_file->get_file_path());

        output_frame_file->write_header();

        std::optional<int> contiguous_frames = std::nullopt;

        frame_buffer = new char[output_frame_size];

        while (nb_frames_to_record_ == std::nullopt ||
               (nb_frames_recorded < nb_frames_to_record_.value() && !stop_requested_))
        {
            if (record_queue.has_overridden() || Holovibes::instance().get_gpu_input_queue()->has_overridden())
            {
                // Due to frames being overwritten when the queue/batchInputQueue is full, the contiguity is lost.
                if (!contiguous_frames.has_value())
                    contiguous_frames = std::make_optional(nb_frames_recorded.load());
            }

            wait_for_frames(record_queue);

            // While wait_for_frames() is running, a stop might be requested and the queue reset.
            // To avoid problems with dequeuing while it's empty, we check right after wait_for_frame
            // and stop recording if needed.
            if (stop_requested_)
                break;

            if (nb_frames_skip_ > 0)
            {
                record_queue.dequeue();
                nb_frames_skip_--;
                continue;
            }

            record_queue.dequeue(frame_buffer, stream_, cudaMemcpyDeviceToHost);
            output_frame_file->write_frame(frame_buffer, output_frame_size);
            (*processed_fps)++;
            nb_frames_recorded++;

            integrate_fps_average();
            if (!nb_frames_to_record_.has_value())
                nb_frames_to_record++;
        }

        LOG_INFO("Recording stopped, written frames : {}", nb_frames_recorded.load());
        output_frame_file->correct_number_of_frames(nb_frames_recorded);

        if (contiguous_frames.has_value())
        {
            LOG_WARN("Record lost its contiguousity at frame {}.", contiguous_frames.value());
            LOG_WARN("To prevent this lost, you might need to increase Input AND/OR Record buffer size.");
        }
        else
        {
            LOG_INFO("Record is contiguous!");
        }

        auto contiguous = contiguous_frames.value_or(nb_frames_recorded);
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

    LOG_TRACE("Exiting FrameRecordWorker::run()");
}

Queue& FrameRecordWorker::init_gpu_record_queue()
{
    auto pipe = Holovibes::instance().get_compute_pipe();
    std::unique_ptr<Queue>& raw_view_queue = pipe->get_raw_view_queue();
    if (raw_view_queue)
        raw_view_queue->resize(4, stream_);

    std::shared_ptr<Queue> output_queue = Holovibes::instance().get_gpu_output_queue();
    if (output_queue)
        output_queue->resize(4, stream_);

    if (record_mode_ == RecordMode::RAW)
    {
        pipe->request_raw_record();
        while (pipe->get_raw_record_requested() && !stop_requested_)
            continue;
    }
    else if (record_mode_ == RecordMode::HOLOGRAM)
    {
        pipe->request_hologram_record();
        while (pipe->get_hologram_record_requested() && !stop_requested_)
            continue;
    }
    else if (record_mode_ == RecordMode::CUTS_YZ || record_mode_ == RecordMode::CUTS_XZ)
    {
        pipe->request_cuts_record(record_mode_);
        while (pipe->get_cuts_record_requested() && !stop_requested_)
            continue;
    }

    return *pipe->get_frame_record_queue();
}

void FrameRecordWorker::wait_for_frames(Queue& record_queue)
{
    auto pipe = Holovibes::instance().get_compute_pipe();
    while (!stop_requested_)
    {
        if (record_queue.get_size() != 0)
            break;
    }
}

void FrameRecordWorker::reset_gpu_record_queue()
{
    auto pipe = Holovibes::instance().get_compute_pipe();
    pipe->request_disable_frame_record();

    std::unique_ptr<Queue>& raw_view_queue = pipe->get_raw_view_queue();
    if (raw_view_queue)
        raw_view_queue->resize(output_buffer_size_, stream_);

    std::shared_ptr<Queue> output_queue = Holovibes::instance().get_gpu_output_queue();
    if (output_queue)
        output_queue->resize(output_buffer_size_, stream_);
}
} // namespace holovibes::worker
