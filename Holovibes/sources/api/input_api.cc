#include "input_api.hh"

#include "API.hh"
#include "camera_exception.hh"
#include "input_frame_file.hh"
#include "input_frame_file_factory.hh"

namespace holovibes::api
{

#pragma region Internals

void InputApi::camera_none() const
{
    api_->compute.close_critical_compute();

    Holovibes::instance().stop_frame_read();

    set_camera_kind_enum(CameraKind::NONE);
    api_->compute.set_is_computation_stopped(true);
    set_import_type(ImportType::None);
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

bool InputApi::import_start() const
{
    LOG_FUNC();

    // Check if computation is currently running
    if (!api_->compute.get_is_computation_stopped())
        import_stop();

    // Because we are in file mode
    camera_none();
    api_->compute.set_is_computation_stopped(false);

    // if the file is to be imported in GPU, we should load the buffer preset for such case
    if (get_load_file_in_gpu())
        NotifierManager::notify<bool>("set_preset_file_gpu", true);

    try
    {
        Holovibes::instance().init_input_queue(get_input_fd(), get_input_buffer_size());
        Holovibes::instance().start_file_frame_read();
    }
    catch (const std::exception& e)
    {
        LOG_ERROR("Catch {}", e.what());
        Holovibes::instance().stop_compute();
        Holovibes::instance().stop_frame_read();
        return false;
    }

    set_import_type(ImportType::File);
    api_->record.set_record_mode(RecordMode::HOLOGRAM);

    return true;
}

void InputApi::import_stop() const
{
    if (get_import_type() == ImportType::None)
        return;

    LOG_FUNC();

    api_->compute.close_critical_compute();

    Holovibes::instance().stop_all_worker_controller();
    Holovibes::instance().start_information_display();

    api_->compute.set_is_computation_stopped(true);
    set_import_type(ImportType::None);
}

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

    // Set settings
    set_input_fd(input->get_frame_descriptor());
    set_input_file_path(filename);
    set_input_file_start_index(0);
    set_input_file_end_index(input->get_total_nb_frames());

    if (!input->get_has_footer())
    {
        if (!json_path.empty())
        {
            LOG_INFO("No footer. Initialization with: {}", json_path);
            API.settings.load_compute_settings(json_path);
        }

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

    return input;
}

#pragma endregion

#pragma region Cameras

bool InputApi::set_camera_kind(CameraKind c, bool save) const
{
    LOG_FUNC(static_cast<int>(c));
    camera_none();

    auto path = holovibes::settings::user_settings_filepath;
    LOG_INFO("path: {}", path);
    std::ifstream input_file(path);
    json j_us = json::parse(input_file);

    if (save)
        j_us["camera"]["type"] = c;

    if (c == CameraKind::NONE)
    {
        if (save)
        {
            std::ofstream output_file(path);
            output_file << j_us.dump(1);
        }

        return true;
    }
    try
    {
        if (api_->compute.get_compute_mode() == Computation::Raw)
            Holovibes::instance().stop_compute();

        set_data_type(RecordedDataType::RAW); // The data gotten from a camera is raw

        try
        {
            Holovibes::instance().start_camera_frame_read(c);
        }
        catch (const std::exception&)
        {
            LOG_INFO("Set camera to NONE");

            if (save)
            {
                j_us["camera"]["type"] = 0;
                std::ofstream output_file(path);
                output_file << j_us.dump(1);
            }

            Holovibes::instance().stop_frame_read();
            return false;
        }

        set_camera_kind_enum(c);
        set_import_type(ImportType::Camera);
        api_->compute.set_is_computation_stopped(false);

        if (save)
        {
            std::ofstream output_file(path);
            output_file << j_us.dump(1);
        }

        return true;
    }
    catch (const camera::CameraException& e)
    {
        LOG_ERROR("[CAMERA] {}", e.what());
    }
    catch (const std::exception& e)
    {
        LOG_ERROR("Catch {}", e.what());
    }

    if (save)
    {
        std::ofstream output_file(path);
        output_file << j_us.dump(1);
    }

    return false;
}

#pragma endregion

} // namespace holovibes::api