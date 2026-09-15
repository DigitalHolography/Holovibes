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
    RecordedDataType data_type = m[setting<settings::RecordMode>()];

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
                                          setting<settings::RecordMode>() == RecordMode::RAW &&
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
    if (setting<settings::RecordMode>() == RecordMode::MOMENTS)
    {
        img_count = total_to_record / 3;
    }

    const size_t output_frame_size = record_queue_.load()->get_fd().get_frame_size();
    const bool direct_ram_staging = async_job_ && setting<settings::RecordMode>() != RecordMode::MOMENTS &&
                                    setting<settings::RecordMode>() != RecordMode::OCT_CUBE &&
                                    setting<settings::RecordMode>() != RecordMode::OCT_CUBE_FLOAT;

    // buffers initialisation
    std::unique_ptr<io_files::OutputFrameFile> owned_output_file;
    io_files::OutputFrameFile* output_frame_file = nullptr;
    char* frame_buffer = nullptr;
    char* moments_buffer = nullptr;
    int moment_idx = 0;

    // Buffer cube for OCT_CUBE / OCT_CUBE_FLOAT
    char* cube_buffer = nullptr;
    size_t cube_size = 0;
    size_t current_cube_slice = 0;
    try
    {
        if (capture_frame_timestamps && img_count > 0)
            frame_timestamps_us.reserve(img_count);
        if (!direct_ram_staging)
            frame_buffer = new char[output_frame_size];
        if (setting<settings::RecordMode>() == RecordMode::MOMENTS)
        {
            moments_buffer = new char[output_frame_size * 3];
        }
        else if (setting<settings::RecordMode>() == RecordMode::OCT_CUBE ||
                 setting<settings::RecordMode>() == RecordMode::OCT_CUBE_FLOAT)
        {
            size_t depth = API.transform.get_time_transformation_size();
            cube_size = depth * output_frame_size;
            cube_buffer = new char[cube_size];
        }
    }
    catch (const std::exception& e)
    {
        LOG_ERROR("Could not allocate recording frame buffers: {}", e.what());
        if (async_job_)
            async_writer_.abort(async_job_, e.what());
        delete[] frame_buffer;
        delete[] moments_buffer;
        delete[] cube_buffer;
        API.record.set_frame_acquisition_enabled(false);
        reset_record_queue();
        FastUpdatesMap::map<IntType>.remove_entry(IntType::SAVING_FPS);
        return;
    }

    while (!API.record.get_frame_acquisition_enabled())
        continue;

    auto write_or_stage_frame = [&](const char* data, size_t size) {
        if (!async_job_)
        {
            output_frame_file->write_frame(data, size);
            return;
        }
        auto copy = std::unique_ptr<char[]>(new char[size]);
        std::memcpy(copy.get(), data, size);
        async_writer_.enqueue(async_job_, std::move(copy), size);
    };

    bool capture_failed = false;
    std::string capture_error;
    try
    {
        owned_output_file.reset(open_output_file(img_count));
        output_frame_file = owned_output_file.get();
        output_frame_file->write_header();
        if (async_job_)
            async_writer_.attach_file(async_job_, std::move(owned_output_file));

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
                if (setting<settings::RecordMode>() == RecordMode::RAW)
                {
                    (void)g_record_stamp_queue.pop_one_blocking(); // consume the corresponding stamp
                }
                nb_frames_to_skip--;
                continue;
            }
            nb_frames_to_skip = setting<settings::FrameSkip>();

            std::unique_ptr<char[]> acquired_frame;
            if (direct_ram_staging)
                acquired_frame = std::unique_ptr<char[]>(new char[output_frame_size]);
            char* dequeue_destination = direct_ram_staging ? acquired_frame.get() : frame_buffer;
            record_queue_.load()->dequeue(dequeue_destination,
                                          stream_,
                                          API.record.get_record_queue_location() == holovibes::Device::GPU
                                              ? cudaMemcpyDeviceToHost
                                              : cudaMemcpyHostToHost);

            uint64_t this_id = 0;
            if (setting<settings::RecordMode>() == RecordMode::RAW)
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
            if (setting<settings::RecordMode>() == RecordMode::MOMENTS)
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
                    write_or_stage_frame(moments_buffer, output_frame_size * 3);
                    (*processed_fps)++;
                    nb_frames_recorded += 3;
                    moment_idx = 0;
                }
            }
            // OCT_CUBE / OCT_CUBE_FLOAT
            else if (setting<settings::RecordMode>() == RecordMode::OCT_CUBE ||
                     setting<settings::RecordMode>() == RecordMode::OCT_CUBE_FLOAT)
            {
                std::memcpy(cube_buffer + current_cube_slice * output_frame_size, frame_buffer, output_frame_size);
                current_cube_slice++;
                (*processed_fps)++;
                nb_frames_recorded++;

                size_t depth = API.transform.get_time_transformation_size();
                if (current_cube_slice == depth)
                {
                    write_or_stage_frame(cube_buffer, cube_size);
                    current_cube_slice = 0;
                }
            }
            // RAW, PROCESSED
            else
            {
                if (direct_ram_staging)
                    async_writer_.enqueue(async_job_, std::move(acquired_frame), output_frame_size);
                else
                    write_or_stage_frame(frame_buffer, output_frame_size);
                if (capture_frame_timestamps)
                    frame_timestamps_us.push_back({last_ts_us, last_camera_ts_us, last_offset_us});
                (*processed_fps)++;
                nb_frames_recorded++;
            }
            integrate_fps_average();
        }

        LOG_INFO("Recording stopped, {} frames {}", nb_frames_recorded.load(), async_job_ ? "buffered" : "written");
        if (!async_job_)
            output_frame_file->correct_number_of_frames(nb_frames_recorded.load());

        if (setting<settings::RecordMode>() == RecordMode::RAW && first_id.has_value())
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

        if (!async_job_)
            output_frame_file->write_footer();
    }
    catch (const std::exception& e)
    {
        LOG_ERROR("{}", e.what());
        capture_failed = true;
        capture_error = e.what();
        API.record.set_frame_acquisition_enabled(false);
        if (auto pipe = API.compute.get_compute_pipe())
            pipe->request(ICS::DisableFrameRecord);
    }

    if (async_job_)
    {
        if (capture_failed)
            async_writer_.abort(async_job_, capture_error);
        else
            async_writer_.finish(async_job_, nb_frames_recorded.load());
    }
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
    if (pipe)
        pipe->request(ICS::DisableFrameRecord);
    g_record_id_queue.clear();
    g_record_stamp_queue.clear();
    record_queue_.load()->reset();
}
} // namespace holovibes::worker
