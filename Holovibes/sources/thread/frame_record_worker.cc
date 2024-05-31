#include "frame_record_worker.hh"
#include "output_frame_file_factory.hh"
#include "tools.hh"
#include "holovibes.hh"
#include "icompute.hh"
#include "global_state_holder.hh"
#include "API.hh"
#include "logger.hh"
#include <spdlog/spdlog.h>
#include <nvtx3/nvToolsExt.h>

namespace holovibes::worker
{
void FrameRecordWorker::integrate_fps_average()
{
    auto& fps_map = GSH::fast_updates_map<FpsType>;
    auto input_fps = fps_map.get_entry(FpsType::INPUT_FPS);
    int current_fps = input_fps->load();

    // Un fps de 0 n'est pas pertinent. Ne pas l'inclure dans la moyenne des fps.
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
    nvtxRangePushA("recording");
    onrestart_settings_.apply_updates();
    LOG_FUNC();

    int nb_frames_to_dequeue = 32;

    auto fast_update_progress_entry = GSH::fast_updates_map<ProgressType>.create_entry(ProgressType::FRAME_RECORD);
    std::atomic<uint>& nb_frames_recorded = fast_update_progress_entry->first;
    std::atomic<uint>& nb_frames_to_record = fast_update_progress_entry->second;

    nb_frames_recorded = 0;
    nb_frames_to_record = setting<settings::RecordFrameCount>().value_or(0);

    std::shared_ptr<std::atomic<uint>> processed_fps = GSH::fast_updates_map<FpsType>.create_entry(FpsType::SAVING_FPS);
    *processed_fps = 0;
    auto pipe = Holovibes::instance().get_compute_pipe();
    pipe->request_frame_record();

    const size_t output_frame_size = record_queue_.load()->get_fd().get_frame_size();
    io_files::OutputFrameFile* output_frame_file = nullptr;
    std::unique_ptr<char[]> frame_buffer(
        new char[output_frame_size * nb_frames_to_dequeue]); // Allocate buffer for multiple frames

    try
    {
        output_frame_file = io_files::OutputFrameFileFactory::create(setting<settings::RecordFilePath>(),
                                                                     record_queue_.load()->get_fd(),
                                                                     nb_frames_to_record);

        LOG_DEBUG("output_frame_file = {}", output_frame_file->get_file_path());

        output_frame_file->write_header();

        std::optional<int> contiguous_frames = std::nullopt;

        size_t nb_frames_to_skip = setting<settings::RecordFrameSkip>();

        if (Holovibes::instance().get_input_queue()->has_overridden())
            Holovibes::instance().get_input_queue()->reset_override();

        while (!setting<settings::RecordFrameCount>() ||
               (nb_frames_recorded < setting<settings::RecordFrameCount>().value() && !stop_requested_))
        {
            if (record_queue_.load()->has_overridden() || Holovibes::instance().get_input_queue()->has_overridden())
            {
                if (!contiguous_frames.has_value())
                    contiguous_frames = std::make_optional(nb_frames_recorded.load());
            }

            wait_for_frames();

            if (stop_requested_)
                break;

            if (nb_frames_to_skip > 0)
            {
                record_queue_.load()->dequeue(); // Skip single frame
                nb_frames_to_skip--;
                continue;
            }

            nvtxRangePushA("dequeue_record");
            int frames_dequeued = record_queue_.load()->dequeue(
                frame_buffer.get(),
                stream_,
                api::get_record_queue_location() == holovibes::Device::GPU ? cudaMemcpyDeviceToHost
                                                                           : cudaMemcpyHostToHost,
                nb_frames_to_dequeue); // Dequeue multiple frames
            nvtxRangePop();

            nvtxRangePushA("write_frame");
            for (int i = 0; i < frames_dequeued; ++i)
            {
                output_frame_file->write_frame(frame_buffer.get() + i * output_frame_size,
                                               output_frame_size); // Write each frame
            }
            nvtxRangePop();

            (*processed_fps) += frames_dequeued;
            nb_frames_recorded += frames_dequeued;

            integrate_fps_average();
            if (!setting<settings::RecordFrameCount>())
                nb_frames_to_record += 32;
        }
        output_frame_file->flush_buffer();
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
        LOG_ERROR("{}", e.what());
    }

    delete output_frame_file;

    reset_record_queue();

    GSH::fast_updates_map<ProgressType>.remove_entry(ProgressType::FRAME_RECORD);
    GSH::fast_updates_map<FpsType>.remove_entry(FpsType::SAVING_FPS);

    LOG_TRACE("Exiting FrameRecordWorker::run()");
    nvtxRangePop();
}

void FrameRecordWorker::wait_for_frames()
{
    nvtxRangePushA("wait_for_frames");
    auto pipe = Holovibes::instance().get_compute_pipe();
    while (!stop_requested_)
    {
        if (record_queue_.load()->get_size() != 0)
            break;
    }
    nvtxRangePop();
}

void FrameRecordWorker::reset_record_queue()
{
    auto pipe = Holovibes::instance().get_compute_pipe();
    pipe->request_disable_frame_record();
    record_queue_.load()->reset();
}
} // namespace holovibes::worker