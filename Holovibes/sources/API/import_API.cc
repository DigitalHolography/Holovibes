#include "API.hh"
#include "logger.hh"

namespace holovibes::api
{

void import_stop()
{
    LOG_FUNC(main);

    close_windows();
    close_critical_compute();

    Holovibes::instance().stop_all_worker_controller();
    Holovibes::instance().start_information_display();

    api::set_import_type(ImportTypeEnum::None);

    set_is_computation_stopped(true);
}

bool import_start(
    std::string& file_path, unsigned int fps, size_t first_frame, bool load_file_in_gpu, size_t last_frame)
{
    LOG_FUNC(main, file_path, fps, first_frame, last_frame, load_file_in_gpu);

    set_is_computation_stopped(false);

    // Because we are in file mode
    UserInterfaceDescriptor::instance().is_enabled_camera_ = false;

    try
    {

        Holovibes::instance().init_input_queue(UserInterfaceDescriptor::instance().file_fd_,
                                               api::get_input_buffer_size());
        Holovibes::instance().start_file_frame_read(file_path,
                                                    true,
                                                    fps,
                                                    static_cast<unsigned int>(first_frame - 1),
                                                    static_cast<unsigned int>(last_frame - first_frame + 1),
                                                    load_file_in_gpu);
    }
    catch (const std::exception& e)
    {
        LOG_ERROR(main, "Catch {}", e.what());
        UserInterfaceDescriptor::instance().is_enabled_camera_ = false;
        Holovibes::instance().stop_compute();
        Holovibes::instance().stop_frame_read();
        return false;
    }

    UserInterfaceDescriptor::instance().is_enabled_camera_ = true;
    api::set_import_type(ImportTypeEnum::File);

    return true;
}

std::optional<io_files::InputFrameFile*> import_file(const std::string& filename)
{
    if (!filename.empty())
    {

        // Will throw if the file format (extension) cannot be handled
        auto input_file = io_files::InputFrameFileFactory::open(filename);

        return input_file;
    }

    return std::nullopt;
}

} // namespace holovibes::api
