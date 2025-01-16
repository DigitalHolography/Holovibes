#include "chrono.hh"
#include "frame_record_worker.hh"
#include "output_frame_file_factory.hh"
#include "tools.hh"
#include "holovibes.hh"
#include "icompute.hh"
#include "fast_updates_holder.hh"
#include "API.hh"
#include "logger.hh"

#include <tuple>
#include <spdlog/spdlog.h>
#include <fstream>
#include <filesystem>

namespace holovibes::worker
{
void FrameRecordWorker::integrate_fps_average()
{
    auto& fps_map = FastUpdatesMap::map<IntType>;
    auto input_fps = fps_map.get_entry(IntType::INPUT_FPS);
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

bool has_input_queue_overwritten()
{
    auto input_queue = API.compute.get_input_queue();
    if (!input_queue)
        return false;

    return input_queue->has_overwritten();
}

io_files::OutputFrameFile* FrameRecordWorker::open_output_file(const uint frame_count)
{
    static std::map<RecordedEyeType, std::string> eye_map{{RecordedEyeType::LEFT, "_L"},
                                                          {RecordedEyeType::NONE, ""},
                                                          {RecordedEyeType::RIGHT, "_R"}};
    // Only add the eye extension if it is the first time recording with it
    std::string eye_string =
        API.input.get_import_type() == ImportType::Camera ? eye_map[setting<settings::RecordedEye>()] : "";

    std::string record_file_path;
    if (Holovibes::instance().is_cli)
        record_file_path = get_record_filename(setting<settings::RecordFilePath>(), eye_string, "R");
    else
        record_file_path = get_record_filename(setting<settings::RecordFilePath>(), eye_string);

    static std::map<RecordMode, RecordedDataType> m = {{RecordMode::RAW, RecordedDataType::RAW},
                                                       {RecordMode::HOLOGRAM, RecordedDataType::PROCESSED},
                                                       {RecordMode::MOMENTS, RecordedDataType::MOMENTS}};
    RecordedDataType data_type = m[API.record.get_record_mode()];

    io_files::OutputFrameFile* output_frame_file =
        io_files::OutputFrameFileFactory::create(record_file_path,
                                                 record_queue_.load()->get_fd(),
                                                 frame_count,
                                                 data_type);

    LOG_DEBUG("output_frame_file = {}", output_frame_file->get_file_path());

    return output_frame_file;
}

bool FrameRecordWorker::all_frames_saved(uint frames_saved, uint total) const
{
    return !API.record.get_frame_acquisition_enabled() && frames_saved >= total;
}

void FrameRecordWorker::run()
{
    onrestart_settings_.apply_updates();
    LOG_FUNC();

    // Progress recording FastUpdatesHolder entry
    auto fast_update_progress_entry = FastUpdatesMap::map<RecordType>.get_or_create_entry(RecordType::FRAME);
    std::atomic<uint>& nb_frames_acquired = std::get<0>(*fast_update_progress_entry);
    std::atomic<uint>& nb_frames_recorded = std::get<1>(*fast_update_progress_entry);
    std::atomic<uint>& nb_frames_to_record = std::get<2>(*fast_update_progress_entry);

    // Processed FPS FastUpdatesHolder entry
    std::shared_ptr<std::atomic<uint>> processed_fps = FastUpdatesMap::map<IntType>.create_entry(IntType::SAVING_FPS);
    *processed_fps = 0;

    size_t nb_frames_to_skip = setting<settings::RecordFrameOffset>();
    auto frame_count = setting<settings::RecordFrameCount>();
    const size_t output_frame_size = record_queue_.load()->get_fd().get_frame_size();

    io_files::OutputFrameFile* output_frame_file = nullptr;
    char* frame_buffer = nullptr;

    while (!API.record.get_frame_acquisition_enabled())
        continue;

    try
    {
        output_frame_file = open_output_file(nb_frames_to_record.load());
        output_frame_file->write_header();

        std::optional<int> contiguous_frames = std::nullopt;

        frame_buffer = new char[output_frame_size];

        while (true)
        {
            if (record_queue_.load()->has_overwritten() || has_input_queue_overwritten())
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

                    if (has_input_queue_overwritten())
                        LOG_WARN("The input queue has been saturated ; the record will stop once all contiguous frames "
                                 "are written");
                }
            }

            // Stop the record when all frames has been aquired and written
            if (all_frames_saved(nb_frames_recorded, nb_frames_to_record))
                break;

            // Stop the record if a queue has overwritten and when all contiguous frames are written
            if (contiguous_frames.has_value() &&
                (std::cmp_greater_equal(nb_frames_recorded.load(), contiguous_frames.value()) ||
                 nb_frames_recorded >= nb_frames_to_record))
                break;

            while (record_queue_.load()->get_size() == 0 && !all_frames_saved(nb_frames_recorded, nb_frames_to_record))
                continue;

            // Stop the record when all frames has been aquired and written
            if (all_frames_saved(nb_frames_recorded, nb_frames_to_record))
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
                                          API.record.get_record_queue_location() == holovibes::Device::GPU
                                              ? cudaMemcpyDeviceToHost
                                              : cudaMemcpyHostToHost);
            output_frame_file->write_frame(frame_buffer, output_frame_size);

            (*processed_fps)++;
            nb_frames_recorded++;

            integrate_fps_average();
        }

        LOG_INFO("Recording stopped, written frames : {}", nb_frames_recorded.load());
        output_frame_file->correct_number_of_frames(nb_frames_recorded);

        if (contiguous_frames.has_value() && std::cmp_less(contiguous_frames.value(), nb_frames_recorded.load()))
        {
            LOG_WARN("Record lost its contiguousity at frame {}.", contiguous_frames.value());
            LOG_WARN("To prevent this lost, you might need to increase Input AND/OR Record buffer size.");
        }
        else
            LOG_INFO("Record is contiguous!");

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

    FastUpdatesMap::map<IntType>.remove_entry(IntType::SAVING_FPS);

    LOG_TRACE("Exiting FrameRecordWorker::run()");
}

void FrameRecordWorker::reset_record_queue()
{
    auto pipe = API.compute.get_compute_pipe();
    pipe->request(ICS::DisableFrameRecord);
    record_queue_.load()->reset();
}
} // namespace holovibes::worker
