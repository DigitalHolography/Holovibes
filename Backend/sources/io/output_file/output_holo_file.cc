#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <chrono>
#include <ctime>
#include <sstream>
#include <iomanip>

#include "output_holo_file.hh"
#include "file_exception.hh"
#include "logger.hh"
#include "holovibes.hh"
#include "API.hh"
#include "camera_config.hh"

namespace holovibes::io_files
{
OutputHoloFile::OutputHoloFile(const std::string& file_path,
                               const camera::FrameDescriptor& fd,
                               uint64_t img_nb,
                               RecordedDataType data_type)
    : OutputFrameFile(file_path)
    , HoloFile()
{
    fd_ = fd;

    holo_file_header_.magic_number[0] = 'H';
    holo_file_header_.magic_number[1] = 'O';
    holo_file_header_.magic_number[2] = 'L';
    holo_file_header_.magic_number[3] = 'O';

    holo_file_header_.version = current_version_;
    holo_file_header_.bits_per_pixel = fd_.depth * camera::PixelDepth::Complex;
    holo_file_header_.img_width = fd_.width;
    holo_file_header_.img_height = fd_.height;
    holo_file_header_.img_nb = static_cast<uint32_t>(img_nb);
    holo_file_header_.endianness = camera::Endianness::LittleEndian;
    holo_file_header_.data_type = static_cast<uint8_t>(data_type);

    holo_file_header_.total_data_size = fd_.get_frame_size() * img_nb;

    meta_data_ = json();

    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm local_tm = *std::localtime(&now_c);

    std::ostringstream ss;
    ss << std::put_time(&local_tm, "%Y-%m-%d %H:%M:%S");
    file_creation_timestamp_ = ss.str();

    prealloc_.io_buf_size = 4u << 20;   // 4 MiB setvbuf
    prealloc_.grow_chunk = 1ull << 30;  // 1 GiB prealloc steps
    prealloc_.durable_on_close = false; // if _commit on close
    prealloc_.attach(file_);            // sets setvbuf; initializes sizes
}

// Optimisation removed to avoid some values in memory to be set to 0 during compilation.
#pragma optimize("", off)

void OutputHoloFile::export_compute_settings(int input_fps, size_t contiguous)
{
    LOG_FUNC(input_fps, contiguous);

    try
    {
        auto& api = API;
        // Determine camera FPS (fallback to input_fps if unavailable)
        int camera_fps = api.input.get_camera_fps() == 0 ? input_fps : api.input.get_camera_fps();
        // Prepare the "camera" object: null if no camera, otherwise with three fields
        nlohmann::json camera_info = nullptr;

        if (api.input.can_get_camera_fps())
        {
            int period = 0;
            int nb_grabbers = 0;
            int exposure_time = 0;
            int buffer_part_count = 0;
            int nb_buffers = 0;
            float gain = 0.00;
            std::string trigger_source = "";

            CameraKind kind = api.input.get_camera_kind();
            if (kind == CameraKind::Phantom || kind == CameraKind::AmetekS711EuresysCoaxlinkQSFP ||
                kind == CameraKind::AmetekS991EuresysCoaxlinkQSFP)
            {
                boost::property_tree::ptree params;
                try
                {
                    boost::property_tree::ini_parser::read_ini(api.input.get_camera_ini_name(), params);
                }
                catch (const boost::property_tree::ini_parser_error& e)
                {
                    LOG_ERROR("Failed to parse camera INI '{}': {}", api.input.get_camera_ini_name(), e.what());
                }
                std::string section;
                switch (kind)
                {
                case CameraKind::Phantom:
                    section = "s710";
                    break;

                case CameraKind::AmetekS711EuresysCoaxlinkQSFP:
                    section = "s711";
                    break;

                case CameraKind::AmetekS991EuresysCoaxlinkQSFP:
                    section = "s991";
                    break;

                default:
                    section = "";
                    break;
                }

                period = params.get<int>(section + ".CycleMinimumPeriod", 0);
                exposure_time = params.get<int>(section + ".ExposureTime", 0);
                nb_grabbers = params.get<int>(section + ".NbGrabbers", 0);
                nb_buffers = params.get<int>(section + ".NbBuffers", 0);
                buffer_part_count = params.get<int>(section + ".BufferPartCount", 0);
                gain = params.get<float>(section + ".Gain", 0);
                trigger_source = params.get<std::string>(section + ".TriggerSource", "");
            }

            camera_info = nlohmann::json{{"Camera_type", api.input.camera_kind_to_string(api.input.get_camera_kind())},
                                         {"CycleMinimumPeriod", period},
                                         {"NbGrabbers", nb_grabbers},
                                         {"TriggerSource", trigger_source},
                                         {"NbBuffers", nb_buffers},
                                         {"BufferPartCount", buffer_part_count},
                                         {"Gain", gain},
                                         {"ExposureTime", exposure_time}};
        }

        // Get current date and time for the record timestamp
        auto now = std::chrono::system_clock::now();
        std::time_t now_c = std::chrono::system_clock::to_time_t(now);
        std::tm local_tm = *std::localtime(&now_c);

        std::ostringstream ss;
        ss << std::put_time(&local_tm, "%Y-%m-%d %H:%M:%S");
        std::string record_timestamp = ss.str();

        // Precise timestamps (us)
        uint64_t first_ts_us = has_session_ts_ ? session_first_ts_us_ : 0;
        uint64_t last_ts_us = has_session_ts_ ? session_last_ts_us_ : 0;
        uint64_t duration_us = (has_session_ts_ && last_ts_us >= first_ts_us) ? (last_ts_us - first_ts_us) : 0;
        uint64_t first_camera_ts_us = has_session_ts_ ? session_first_camera_ts_us_ : 0;
        uint64_t last_camera_ts_us = has_session_ts_ ? session_last_camera_ts_us_ : 0;
        uint64_t first_offset_us = has_session_ts_ ? session_first_offset_us_ : 0;
        uint64_t last_offset_us = has_session_ts_ ? session_last_offset_us_ : 0;

        // Build the info JSON without top-level camera_fps
        auto j_fi =
            nlohmann::json{{"pixel_pitch", {{"x", api.input.get_pixel_size()}, {"y", api.input.get_pixel_size()}}},
                           {"input_fps", api.input.can_get_camera_fps() ? camera_fps : input_fps},
                           {"camera_fps", camera_fps}, // camera frames per second
                           {"eye_type", api.record.get_recorded_eye()},
                           {"contiguous", contiguous},
                           {"holovibes_version", __HOLOVIBES_VERSION__},
                           {"camera", camera_info},
                           {"file_create_timestamp", file_creation_timestamp_},
                           {"file_record_timestamp", record_timestamp},
                           {"timestamps_us",
                            {{"unix_first", first_ts_us},
                             {"unix_last", last_ts_us},
                             {"duration", duration_us},
                             {"camera_first", first_camera_ts_us},
                             {"camera_last", last_camera_ts_us},
                             {"offset_first", first_offset_us},
                             {"offset_last", last_offset_us}}}};

        meta_data_ = nlohmann::json{{"compute_settings", api.settings.compute_settings_to_json()}, {"info", j_fi}};
    }
    catch (const nlohmann::json::exception& e)
    {
        meta_data_ = nlohmann::json();
        LOG_WARN("An error was encountered while trying to export compute settings");
        LOG_WARN("Exception: {}", e.what());
    }
}

#pragma optimize("", on)

void OutputHoloFile::write_header()
{
    if (std::fwrite(&holo_file_header_, 1, sizeof(HoloFileHeader), file_) != sizeof(HoloFileHeader))
        throw FileException("Unable to write output holo file header");
}

size_t OutputHoloFile::write_frame(const char* frame, size_t frame_size)
{
    // Grow file in big NTFS chunks when needed, using current logical end.
    prealloc_.ensure_capacity(file_, prealloc_.logical_size + frame_size);

    const size_t written_bytes = std::fwrite(frame, 1, frame_size, file_);
    if (written_bytes != frame_size)
        throw FileException("Unable to write output holo file frame");

    // Track logical file growth
    prealloc_.on_bytes_written(written_bytes);
    return written_bytes;
}

void OutputHoloFile::write_footer()
{
    LOG_FUNC();

    try
    {
        std::string meta_data_str = meta_data_.dump();

        // Ensure capacity for footer at EOF, then write it
        prealloc_.ensure_capacity(file_, prealloc_.logical_size + meta_data_str.size());

        if (std::fwrite(meta_data_str.data(), 1, meta_data_str.size(), file_) != meta_data_str.size())
            throw FileException("Unable to write output holo file footer");

        prealloc_.on_bytes_written(meta_data_str.size());

        // Flush stdio, trim any over-allocation to exact logical size,
        // and (optionally) commit to disk (durable_on_close).
        prealloc_.finalize(file_);
    }
    catch (const std::exception& e)
    {
        LOG_ERROR("Catch {}", e.what());
        throw;
    }
}

void OutputHoloFile::correct_number_of_frames(size_t nb_frames_written)
{
    fpos_t previous_pos;

    if (std::fgetpos(file_, &previous_pos))
        throw FileException("Unable to correct number of written frames");

    holo_file_header_.img_nb = static_cast<uint32_t>(nb_frames_written);
    holo_file_header_.total_data_size = fd_.get_frame_size() * nb_frames_written;

    fpos_t file_begin_pos = 0;

    if (std::fsetpos(file_, &file_begin_pos))
        throw FileException("Unable to correct number of written frames");

    write_header();

    if (std::fsetpos(file_, &previous_pos))
        throw FileException("Unable to correct number of written frames");
}
} // namespace holovibes::io_files
