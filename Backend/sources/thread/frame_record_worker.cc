#include "chrono.hh"
#include "frame_record_worker.hh"
#include "output_frame_file_factory.hh"
#include "tools.hh"
#include "holovibes.hh"
#include "icompute.hh"
#include "fast_updates_holder.hh"
#include "API.hh"
#include "logger.hh"
#include "time_map.hh"
#include "id_queue.hh"
#include "stamp_queue.hh"
#include "output_holo_file.hh"

#include <tuple>
#include <spdlog/spdlog.h>
#include <fstream>
#include <filesystem>

extern FrameTimeMap g_time_map;
extern IdQueue g_record_id_queue;
extern StampQueue g_record_stamp_queue;
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
    if (setting<settings::IsCli>())
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

    std::optional<uint64_t> first_id;
    uint64_t last_id = 0;
    uint64_t first_ts_us = 0, last_ts_us = 0;
    uint64_t first_camera_ts_us = 0, last_camera_ts_us = 0;
    uint64_t first_offset_us = 0, last_offset_us = 0;
    const bool capture_frame_timestamps = setting<settings::RecordFrameTimestampsEnabled>() &&
                                          API.record.get_record_mode() == RecordMode::RAW &&
                                          API.input.get_import_type() == ImportType::Camera;
    std::vector<io_files::FrameTimestampUs> frame_timestamps_us;

    auto fast_update_progress_entry = FastUpdatesMap::map<RecordType>.get_or_create_entry(RecordType::FRAME);
    std::atomic<uint>& nb_frames_acquired = std::get<0>(*fast_update_progress_entry);
    std::atomic<uint>& nb_frames_recorded = std::get<1>(*fast_update_progress_entry);
    std::atomic<uint>& nb_frames_to_record = std::get<2>(*fast_update_progress_entry);

    auto processed_fps = FastUpdatesMap::map<IntType>.create_entry(IntType::SAVING_FPS);
    *processed_fps = 0;

    size_t nb_frames_to_skip = setting<settings::RecordFrameOffset>();
    uint total_to_record = nb_frames_to_record.load();

    // for MOMENTS, 3 plans = 1 frame
    uint img_count = total_to_record;
    if (API.record.get_record_mode() == RecordMode::MOMENTS)
    {
        img_count = total_to_record / 3;
    }

    if (capture_frame_timestamps && img_count > 0)
        frame_timestamps_us.reserve(img_count);

    const size_t output_frame_size = record_queue_.load()->get_fd().get_frame_size();

    // buffers initialisation
    io_files::OutputFrameFile* output_frame_file = nullptr;
    char* frame_buffer = new char[output_frame_size];
    char* moments_buffer = nullptr;
    int moment_idx = 0;

    // Buffer cube for OCT_CUBE / OCT_CUBE_FLOAT
    char* cube_buffer = nullptr;
    size_t cube_size = 0;
    size_t current_cube_slice = 0;
    if (API.record.get_record_mode() == RecordMode::MOMENTS)
    {
        moments_buffer = new char[output_frame_size * 3];
    }
    else if (API.record.get_record_mode() == RecordMode::OCT_CUBE ||
             API.record.get_record_mode() == RecordMode::OCT_CUBE_FLOAT)
    {
        size_t depth = API.transform.get_time_transformation_size();
        cube_size = depth * output_frame_size;
        cube_buffer = new char[cube_size];
    }

    while (!API.record.get_frame_acquisition_enabled())
        continue;

    try
    {
        output_frame_file = open_output_file(img_count);
        output_frame_file->write_header();

        std::optional<int> contiguous_frames = std::nullopt;

        while (true)
        {
            if (!API.record.get_frame_acquisition_enabled() && nb_frames_recorded.load() >= nb_frames_to_record.load())
                break;

            while (record_queue_.load()->get_size() == 0 && (API.record.get_frame_acquisition_enabled() ||
                                                             nb_frames_recorded.load() < nb_frames_to_record.load()))
                continue;

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

            // Skip initial frames

            if (nb_frames_to_skip > 0)
            {
                record_queue_.load()->dequeue();
                if (API.record.get_record_mode() == RecordMode::RAW)
                {
                    (void)g_record_stamp_queue.pop_one_blocking(); // consume the corresponding stamp
                }
                nb_frames_to_skip--;
                continue;
            }
            nb_frames_to_skip = setting<settings::FrameSkip>();

            record_queue_.load()->dequeue(frame_buffer,
                                          stream_,
                                          API.record.get_record_queue_location() == holovibes::Device::GPU
                                              ? cudaMemcpyDeviceToHost
                                              : cudaMemcpyHostToHost);

            uint64_t this_id = 0;
            if (API.record.get_record_mode() == RecordMode::RAW)
            {
                const FrameStamp st = g_record_stamp_queue.pop_one_blocking();
                this_id = st.id;
                const uint64_t this_ts = st.synced_us;
                if (capture_frame_timestamps && frame_timestamps_us.empty() && this_ts == 0)
                    LOG_WARN("The current camera does not provide frame timestamps; per-frame footer values will be "
                             "zero");
                if (!first_id)
                {
                    first_id = this_id;
                    first_ts_us = this_ts;
                    first_camera_ts_us = st.camera_us;
                    first_offset_us = st.offset_us;
                }
                last_id = this_id;
                last_ts_us = this_ts;
                last_camera_ts_us = st.camera_us;
                last_offset_us = st.offset_us;
                // LOG_ERROR(this_ts);
                // LOG_ERROR(this_id);
            }

            // MOMENTS
            if (API.record.get_record_mode() == RecordMode::MOMENTS)
            {
                auto in_f = reinterpret_cast<float*>(frame_buffer);
                auto out_f = reinterpret_cast<float*>(moments_buffer);
                size_t npix = output_frame_size / sizeof(float); // # pixels = H×W
                for (size_t i = 0; i < npix; ++i)
                {
                    out_f[i * 3 + moment_idx] = in_f[i];
                }
                moment_idx++;

                if (moment_idx == 3)
                {
                    // [H×W×3]
                    output_frame_file->write_frame(moments_buffer, output_frame_size * 3);
                    (*processed_fps)++;
                    nb_frames_recorded += 3;
                    moment_idx = 0;
                }
            }
            // OCT_CUBE / OCT_CUBE_FLOAT
            else if (API.record.get_record_mode() == RecordMode::OCT_CUBE ||
                     API.record.get_record_mode() == RecordMode::OCT_CUBE_FLOAT)
            {
                std::memcpy(cube_buffer + current_cube_slice * output_frame_size, frame_buffer, output_frame_size);
                current_cube_slice++;
                (*processed_fps)++;
                nb_frames_recorded++;

                size_t depth = API.transform.get_time_transformation_size();
                if (current_cube_slice == depth)
                {
                    output_frame_file->write_frame(cube_buffer, cube_size);
                    current_cube_slice = 0;
                }
            }
            // RAW, PROCESSED
            else
            {
                output_frame_file->write_frame(frame_buffer, output_frame_size);
                if (capture_frame_timestamps)
                    frame_timestamps_us.push_back({last_ts_us, last_camera_ts_us, last_offset_us});
                (*processed_fps)++;
                nb_frames_recorded++;
            }
            integrate_fps_average();
        }

        LOG_INFO("Recording stopped, written frames: {}", nb_frames_recorded.load());
        output_frame_file->correct_number_of_frames(nb_frames_recorded.load());

        if (API.record.get_record_mode() == RecordMode::RAW && first_id.has_value())
        {
            LOG_INFO("Record timestamps (us): first={} last={}", first_ts_us, last_ts_us);

            const uint64_t duration_us = (last_ts_us >= first_ts_us) ? (last_ts_us - first_ts_us) : 0;

            LOG_INFO("Record duration: {} us ({} ms, {:.3f} s)",
                     duration_us,
                     duration_us / 1000,
                     static_cast<double>(duration_us) / 1'000'000.0);
            if (auto* holo = dynamic_cast<io_files::OutputHoloFile*>(output_frame_file))
            {
                holo->set_session_timestamps_us(first_ts_us,
                                                last_ts_us,
                                                first_camera_ts_us,
                                                last_camera_ts_us,
                                                first_offset_us,
                                                last_offset_us);
                if (!frame_timestamps_us.empty())
                    holo->set_frame_timestamps_us(std::move(frame_timestamps_us));
            }
        }

        if (contiguous_frames.has_value() && std::cmp_less(contiguous_frames.value(), nb_frames_recorded.load()))
        {
            LOG_WARN("Record lost its contiguousity at frame {}.", contiguous_frames.value());
            LOG_WARN("To prevent this lost, you might need to increase Input AND/OR Record buffer size.");
        }
        else
            LOG_INFO("Record is contiguous!");

        size_t contiguous = contiguous_frames.value_or(nb_frames_recorded.load());
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
    if (moments_buffer)
        delete[] moments_buffer;
    if (cube_buffer)
        delete[] cube_buffer;

    reset_record_queue();
    FastUpdatesMap::map<IntType>.remove_entry(IntType::SAVING_FPS);

    LOG_TRACE("Exiting FrameRecordWorker::run()");
}

void FrameRecordWorker::reset_record_queue()
{
    auto pipe = API.compute.get_compute_pipe();
    pipe->request(ICS::DisableFrameRecord);
    g_record_id_queue.clear();
    g_record_stamp_queue.clear();
    record_queue_.load()->reset();
}
} // namespace holovibes::worker
