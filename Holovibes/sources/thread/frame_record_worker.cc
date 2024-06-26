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
    onrestart_settings_.apply_updates();
    LOG_FUNC();
    // Progress recording FastUpdatesHolder entry

    auto fast_update_progress_entry = GSH::fast_updates_map<ProgressType>.create_entry(ProgressType::FRAME_RECORD);
    std::atomic<uint>& nb_frames_recorded = fast_update_progress_entry->first;
    std::atomic<uint>& nb_frames_to_record = fast_update_progress_entry->second;

    nb_frames_recorded = 0;

    if (setting<settings::RecordFrameCount>().has_value())
    {
        nb_frames_to_record = setting<settings::RecordFrameCount>().value();
    }
    else
        nb_frames_to_record = 0;

    // Processed FPS FastUpdatesHolder entry

    std::shared_ptr<std::atomic<uint>> processed_fps = GSH::fast_updates_map<FpsType>.create_entry(FpsType::SAVING_FPS);
    *processed_fps = 0;
    auto pipe = Holovibes::instance().get_compute_pipe();
    pipe->request_frame_record();
    // Queue& record_queue = *pipe->get_frame_record_queue();

    const size_t output_frame_size = record_queue_.load()->get_fd().get_frame_size();
    io_files::OutputFrameFile* output_frame_file = nullptr;
    char* frame_buffer = nullptr;

    try
    {
        output_frame_file = io_files::OutputFrameFileFactory::create(setting<settings::RecordFilePath>(),
                                                                     record_queue_.load()->get_fd(),
                                                                     nb_frames_to_record);

        LOG_DEBUG("output_frame_file = {}", output_frame_file->get_file_path());

        output_frame_file->write_header();

        std::optional<int> contiguous_frames = std::nullopt;

        frame_buffer = new char[output_frame_size];

        size_t nb_frames_to_skip = setting<settings::RecordFrameSkip>();

        if (Holovibes::instance().get_input_queue()->has_overwritten())
            Holovibes::instance().get_input_queue()->reset_override();

        while (setting<settings::RecordFrameCount>() == std::nullopt ||
               (nb_frames_recorded < setting<settings::RecordFrameCount>().value() && !stop_requested_))
        {
            if (record_queue_.load()->has_overwritten() || Holovibes::instance().get_input_queue()->has_overwritten())
            {
                // Due to frames being overwritten when the queue/batchInputQueue is full, the contiguity is lost.
                if (!contiguous_frames.has_value())
                {

                    contiguous_frames =
                        std::make_optional(nb_frames_recorded.load() + record_queue_.load()->get_size());

                    if (record_queue_.load()->has_overwritten())
                        LOG_WARN(
                            "The record queue has been saturated ; the record will stop once all contiguous frames "
                            "are written");

                    if (Holovibes::instance().get_input_queue()->has_overwritten())
                        LOG_WARN("The input queue has been saturated ; the record will stop once all contiguous frames "
                                 "are written");
                }
            }

            wait_for_frames();

            // While wait_for_frames() is running, a stop might be requested and the queue reset.
            // To avoid problems with dequeuing while it's empty, we check right after wait_for_frame
            // and stop recording if needed.
            if (stop_requested_ || (contiguous_frames.has_value() && nb_frames_recorded >= contiguous_frames.value()))
                break;

            if (nb_frames_to_skip > 0)
            {
                record_queue_.load()->dequeue();
                nb_frames_to_skip--;
                continue;
            }

            record_queue_.load()->dequeue(frame_buffer,
                                          stream_,
                                          api::get_record_queue_location() == holovibes::Device::GPU
                                              ? cudaMemcpyDeviceToHost
                                              : cudaMemcpyHostToHost);
            output_frame_file->write_frame(frame_buffer, output_frame_size);

            // FIXME: to check if it's still relevant
            // if (api::get_record_queue_location()) {
            //     record_queue_.load()->dequeue(frame_buffer, stream_, cudaMemcpyDeviceToHost);
            //     output_frame_file->write_frame(frame_buffer, output_frame_size);
            // }
            // else
            // {
            //     {
            //         MutexGuard mGuard(record_queue_.load()->get_guard());
            //         output_frame_file->write_frame(static_cast<char*>(record_queue_.load()->get_data()),
            //         record_queue_.load()->get_size() * output_frame_size);
            //     }
            //     record_queue_.load()->dequeue(record_queue_.load()->get_size());
            // }
            (*processed_fps)++;
            nb_frames_recorded++;

            integrate_fps_average();
            if (!setting<settings::RecordFrameCount>().has_value())
                nb_frames_to_record++;
        }

        // api::set_record_frame_skip(nb_frames_to_skip);

        // api::set_record_frame_skip(nb_frames_to_skip);
        LOG_INFO("Recording stopped, written frames : {}", nb_frames_recorded.load());
        output_frame_file->correct_number_of_frames(nb_frames_recorded);

        if (contiguous_frames.has_value() && contiguous_frames < nb_frames_recorded)
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
    delete[] frame_buffer;

    reset_record_queue();

    GSH::fast_updates_map<ProgressType>.remove_entry(ProgressType::FRAME_RECORD);
    GSH::fast_updates_map<FpsType>.remove_entry(FpsType::SAVING_FPS);

    LOG_TRACE("Exiting FrameRecordWorker::run()");
}

void FrameRecordWorker::wait_for_frames()
{
    auto pipe = Holovibes::instance().get_compute_pipe();
    while (!stop_requested_)
    {
        if (record_queue_.load()->get_size() != 0)
            break;
    }
}

void FrameRecordWorker::reset_record_queue()
{
    auto pipe = Holovibes::instance().get_compute_pipe();
    pipe->request_disable_frame_record();
    record_queue_.load()->reset();

    /*std::unique_ptr<Queue>& raw_view_queue = pipe->get_raw_view_queue();
    if (raw_view_queue)
        raw_view_queue->resize(setting<settings::OutputBufferSize>(), stream_);

    std::shared_ptr<Queue> output_queue = Holovibes::instance().get_gpu_output_queue();
    if (output_queue)
        output_queue->resize(setting<settings::OutputBufferSize>(), stream_);*/
}
} // namespace holovibes::worker
