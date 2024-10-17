#include "chrono.hh"
#include "frame_record_worker.hh"
#include "output_frame_file_factory.hh"
#include "tools.hh"
#include "holovibes.hh"
#include "icompute.hh"
#include "fast_updates_holder.hh"
#include "API.hh"
#include "logger.hh"
#include <spdlog/spdlog.h>
#include <fstream>
#include <filesystem>

namespace holovibes::worker
{
void FrameRecordWorker::integrate_fps_average()
{
    auto& fps_map = FastUpdatesMap::map<FpsType>;
    auto input_fps = fps_map.get_entry(FpsType::INPUT_FPS);
    int current_fps = input_fps->load();

    // An fps of 0 is not relevent. We do not includ it in fps average.
    if (current_fps == 0)
        return;

    fps_buffer_[fps_current_index_++ % FPS_LAST_X_VALUES] = current_fps;
}

size_t FrameRecordWorker::compute_fps_average() const
{
    LOG_TRACE("fps_current_index_ = {}", fps_current_index_);

    if (fps_current_index_ == 0)
        return 0;

    size_t ret = 0;
    size_t upper = FPS_LAST_X_VALUES < fps_current_index_ ? FPS_LAST_X_VALUES : fps_current_index_;
    for (size_t i = 0; i < upper; i++)
        ret += fps_buffer_[i];

    ret /= upper;

    return ret;
}

std::string append_date_to_filepath(std::string record_file_path)
{
    //? Do we move this to the export panel, and consider the date to be set when the path is set/on startup ?
    std::filesystem::path filePath(record_file_path);
    std::string date = Chrono::get_current_date();
    std::string filename = filePath.filename().string();
    std::string path = filePath.parent_path().string();
    std::filesystem::path newFilePath = path + "/" + date + "_" + filename;
    return newFilePath.string();
}

void FrameRecordWorker::run()
{
    onrestart_settings_.apply_updates();
    LOG_FUNC();
    // Progress recording FastUpdatesHolder entry

    auto fast_update_progress_entry = FastUpdatesMap::map<ProgressType>.create_entry(ProgressType::FRAME_RECORD);
    std::atomic<uint>& nb_frames_recorded = fast_update_progress_entry->first;
    std::atomic<uint>& nb_frames_to_record = fast_update_progress_entry->second;

    size_t nb_frames_to_skip = setting<settings::RecordFrameSkip>();

    nb_frames_recorded = 0;

    // Get the real number of frames to record taking in account the frame skip
    auto frame_count = setting<settings::RecordFrameCount>();
    if (frame_count.has_value())
    {
        nb_frames_to_record = static_cast<unsigned int>(frame_count.value());
        nb_frames_to_record = std::ceil((float)(nb_frames_to_record) / (float)(setting<settings::FrameSkip>() + 1));
        // One frame will result in three moments.
        if (setting<settings::RecordMode>() == RecordMode::MOMENTS)
            nb_frames_to_record = nb_frames_to_record * 3;
    }
    else
        nb_frames_to_record = 0;
    // Processed FPS FastUpdatesHolder entry

    std::shared_ptr<std::atomic<uint>> processed_fps = FastUpdatesMap::map<FpsType>.create_entry(FpsType::SAVING_FPS);
    *processed_fps = 0;
    auto pipe = Holovibes::instance().get_compute_pipe();
    pipe->request(ICS::FrameRecord);

    const size_t output_frame_size = record_queue_.load()->get_fd().get_frame_size();
    io_files::OutputFrameFile* output_frame_file = nullptr;
    char* frame_buffer = nullptr;

    try
    {
        std::string record_file_path = append_date_to_filepath(setting<settings::RecordFilePath>());

        output_frame_file = io_files::OutputFrameFileFactory::create(record_file_path,
                                                                     record_queue_.load()->get_fd(),
                                                                     nb_frames_to_record);

        LOG_DEBUG("output_frame_file = {}", output_frame_file->get_file_path());

        output_frame_file->write_header();

        std::optional<int> contiguous_frames = std::nullopt;

        frame_buffer = new char[output_frame_size];

        auto input_queue = api::get_input_queue();

        if (input_queue->has_overwritten())
            input_queue->reset_override();

        while (!stop_requested_ && (frame_count == std::nullopt || nb_frames_recorded < nb_frames_to_record))
        {
            if (record_queue_.load()->has_overwritten() || input_queue->has_overwritten())
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

                    if (input_queue->has_overwritten())
                        LOG_WARN("The input queue has been saturated ; the record will stop once all contiguous frames "
                                 "are written");
                }
            }

            wait_for_frames();

            // While wait_for_frames() is running, a stop might be requested and the queue reset.
            // To avoid problems with dequeuing while it's empty, we check right after wait_for_frame
            // and stop recording if needed.
            if (stop_requested_ || (contiguous_frames.has_value() &&
                                    std::cmp_greater_equal(nb_frames_recorded.load(), contiguous_frames.value())))
                break;

            if (nb_frames_to_skip > 0)
            {
                record_queue_.load()->dequeue();
                nb_frames_to_skip--;
                continue;
            }
            nb_frames_to_skip = setting<settings::FrameSkip>();

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

        if (contiguous_frames.has_value() && std::cmp_less(contiguous_frames.value(), nb_frames_recorded.load()))
        {
            LOG_WARN("Record lost its contiguousity at frame {}.", contiguous_frames.value());
            LOG_WARN("To prevent this lost, you might need to increase Input AND/OR Record buffer size.");
        }
        else
        {
            LOG_INFO("Record is contiguous!");
        }

        auto contiguous = contiguous_frames.value_or(nb_frames_recorded);
        // Change the fps according to the frame skip
        output_frame_file->export_compute_settings(
            static_cast<int>(compute_fps_average() / (setting<settings::FrameSkip>() + 1)),
            contiguous);

        output_frame_file->write_footer();
    }
    catch (const io_files::FileException& e)
    {
        LOG_ERROR("{}", e.what());
    }

    delete output_frame_file;
    delete[] frame_buffer;

    reset_record_queue();

    FastUpdatesMap::map<ProgressType>.remove_entry(ProgressType::FRAME_RECORD);
    FastUpdatesMap::map<FpsType>.remove_entry(FpsType::SAVING_FPS);

    LOG_TRACE("Exiting FrameRecordWorker::run()");
}

void FrameRecordWorker::wait_for_frames()
{
    while (!stop_requested_ && record_queue_.load()->get_size() == 0)
        continue;
}

void FrameRecordWorker::reset_record_queue()
{
    auto pipe = Holovibes::instance().get_compute_pipe();
    pipe->request(ICS::DisableFrameRecord);
    record_queue_.load()->reset();

    /*std::unique_ptr<Queue>& raw_view_queue = pipe->get_raw_view_queue();
    if (raw_view_queue)
        raw_view_queue->resize(setting<settings::OutputBufferSize>(), stream_);

    std::shared_ptr<Queue> output_queue = Holovibes::instance().get_gpu_output_queue();
    if (output_queue)
        output_queue->resize(setting<settings::OutputBufferSize>(), stream_);*/
}
} // namespace holovibes::worker
