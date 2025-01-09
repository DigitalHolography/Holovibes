#include "input_api.hh"

#include "API.hh"
#include "camera_exception.hh"
#include "camera_dll.hh"
#include "input_frame_file.hh"
#include "input_frame_file_factory.hh"
#include "notifier.hh"

namespace holovibes::api
{

#pragma region Internals

void InputApi::camera_none() const
{
    api_->compute.stop();

    set_camera_kind_enum(CameraKind::NONE);
    set_import_type(ImportType::None);

    // Close camera and free ressources
    if (Holovibes::instance().active_camera_)
        Holovibes::instance().active_camera_->shutdown_camera();
    Holovibes::instance().active_camera_.reset();
}

#pragma endregion

#pragma region File Offest

void InputApi::set_input_file_start_index(size_t value) const
{
    // Ensures that moments are read 3 by 3
    if (get_data_type() == RecordedDataType::MOMENTS)
        value -= value % 3;

    UPDATE_SETTING(InputFileStartIndex, value);
    if (value >= get_input_file_end_index())
        set_input_file_end_index(value + 1);
}

void InputApi::set_input_file_end_index(size_t value) const
{
    if (value <= 0)
        value = 1; // Cannot read no frames, so end index can't be 0
    // Ensures that moments are read 3 by 3, and at least 3 (hence the value < 3 check)
    // In moments mode, the end index must at least be
    uchar mod = value % 3;
    if (get_data_type() == RecordedDataType::MOMENTS && (mod != 0 || value < 3))
        value += 3 - mod; // 'ceiling' moments

    UPDATE_SETTING(InputFileEndIndex, value);
    if (value <= get_input_file_start_index())
        set_input_file_start_index(value - 1);
}

#pragma endregion

#pragma region File Import

std::optional<io_files::InputFrameFile*> InputApi::import_file(const std::string& filename,
                                                               const std::string& json_path) const
{
    if (filename.empty())
    {
        LOG_ERROR("Empty filename");
        return std::nullopt;
    }

    io_files::InputFrameFile* input = nullptr;

    // Try to open the file
    try
    {
        input = io_files::InputFrameFileFactory::open(filename);
    }
    catch (const io_files::FileException& e)
    {
        LOG_ERROR("Catch {}", e.what());
        return std::nullopt;
    }

    // Stop any computation currently running
    camera_none();

    // Set settings
    set_input_fd(input->get_frame_descriptor());
    set_input_file_path(filename);
    set_input_file_start_index(0);
    set_input_file_end_index(input->get_total_nb_frames());
    set_import_type(ImportType::File);
    api_->record.set_record_mode(RecordMode::HOLOGRAM);

    if (!input->get_has_footer())
    {
        if (!json_path.empty())
        {
            LOG_INFO("No footer. Initialization with: {}", json_path);
            API.settings.load_compute_settings(json_path);
        }

        if (get_file_load_kind() != FileLoadKind::REGULAR)
            NotifierManager::notify<bool>("set_preset_file_gpu", true);

        return input;
    }

    // Get the buffer size that will be used to allocate the buffer for reading the file instead of the one from the
    // record
    auto input_buffer_size = get_input_buffer_size();
    auto record_buffer_size = api_->record.get_record_buffer_size();

    // Try importing the compute settings from the file. If it fails, we will use the default values
    try
    {
        input->import_compute_settings();
        input->import_info();

        // When reading moments, the batch size must be forced to 3. Because they are 3 moments per frame.
        if (get_data_type() == RecordedDataType::MOMENTS)
        {
            api_->transform.set_batch_size(3);
            api_->transform.set_time_stride(3);
        }
    }
    catch (const std::exception& e)
    {
        LOG_ERROR("Catch {}", e.what());

        if (!json_path.empty())
        {
            LOG_INFO("Compute settings incorrect or file not found. Initialization with: {}", json_path);
            API.settings.load_compute_settings(json_path);
        }
    }

    // update the buffer size with the old values to avoid surcharging the gpu memory in case of big
    // buffers used when the file was recorded
    set_input_buffer_size(input_buffer_size);
    api_->record.set_record_buffer_size(record_buffer_size);

    if (get_file_load_kind() != FileLoadKind::REGULAR)
        NotifierManager::notify<bool>("set_preset_file_gpu", true);

    return input;
}

#pragma endregion

#pragma region Cameras

bool InputApi::set_camera_kind(CameraKind c, bool save) const
{
    camera_none();

    auto path = holovibes::settings::user_settings_filepath;
    LOG_INFO("path: {}", path);
    std::ifstream input_file(path);
    json j_us = json::parse(input_file);

    if (save)
    {
        j_us["camera"]["type"] = c;
        std::ofstream output_file(path);
        output_file << j_us.dump(1);
    }

    if (c == CameraKind::NONE)
        return true;

    try
    {
        const static std::map<CameraKind, LPCSTR> camera_dictionary = {
            {CameraKind::Adimec, "CameraAdimec.dll"},
            {CameraKind::BitflowCyton, "BitflowCyton.dll"},
            {CameraKind::IDS, "CameraIds.dll"},
            {CameraKind::Hamamatsu, "CameraHamamatsu.dll"},
            {CameraKind::xiQ, "CameraXiq.dll"},
            {CameraKind::xiB, "CameraXib.dll"},
            {CameraKind::OpenCV, "CameraOpenCV.dll"},
            {CameraKind::Phantom, "AmetekS710EuresysCoaxlinkOcto.dll"},
            {CameraKind::AmetekS711EuresysCoaxlinkQSFP, "AmetekS711EuresysCoaxlinkQsfp+.dll"},
            {CameraKind::AmetekS991EuresysCoaxlinkQSFP, "AmetekS991EuresysCoaxlinkQsfp+.dll"},
            {CameraKind::Ametek, "EuresyseGrabber.dll"},
            {CameraKind::Alvium, "CameraAlvium.dll"},
            {CameraKind::AutoDetectionPhantom, "CameraPhantomAutoDetection.dll"},
        };

        // Load the camera
        auto active_camera = camera::CameraDLL::load_camera(camera_dictionary.at(c));
        Holovibes::instance().active_camera_ = active_camera;

        set_pixel_size(active_camera->get_pixel_size());
        set_import_type(ImportType::Camera);
        set_input_fd(active_camera->get_fd());
        set_camera_kind_enum(c);
        set_data_type(RecordedDataType::RAW);
    }
    catch (const std::exception& e)
    {
        LOG_ERROR("Catch {}", e.what());
        LOG_INFO("Set camera to NONE");

        if (save)
        {
            j_us["camera"]["type"] = 0;
            std::ofstream output_file(path);
            output_file << j_us.dump(1);
        }

        camera_none();
        Holovibes::instance().stop_frame_read();
        return false;
    }

    return true;
}

#pragma endregion

} // namespace holovibes::api