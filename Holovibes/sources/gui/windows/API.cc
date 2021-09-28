#include "api.hh"

#include <optional>
namespace holovibes::api
{

std::optional<::holovibes::io_files::InputFrameFile*> import_file(const std::string& filename)
{
    LOG_INFO;

    if (!filename.empty())
    {

        // Will throw if the file format (extension) cannot be handled
        auto input_file = ::holovibes::io_files::InputFrameFileFactory::open(filename);

        return input_file;
    }

    return std::nullopt;
}

bool init_holovibes_import_mode(Holovibes& holovibes,
                                camera::FrameDescriptor& file_fd,
                                bool& is_enabled_camera,
                                std::string& file_path,
                                unsigned int fps,
                                size_t first_frame,
                                bool load_file_in_gpu,
                                size_t last_frame)
{
    LOG_INFO;

    // Set the image rendering ui params
    holovibes.get_cd().time_transformation_stride = std::ceil(static_cast<float>(fps) / 20.0f);
    holovibes.get_cd().batch_size = 1;

    // Because we are in import mode
    is_enabled_camera = false;

    try
    {

        holovibes.init_input_queue(file_fd);
        holovibes.start_file_frame_read(file_path,
                                        true,
                                        fps,
                                        first_frame - 1,
                                        last_frame - first_frame + 1,
                                        load_file_in_gpu,
                                        [=]() { return; });
    }
    catch (const std::exception& e)
    {
        LOG_ERROR << e.what();
        is_enabled_camera = false;
        holovibes.stop_compute();
        holovibes.stop_frame_read();
        return false;
    }
    is_enabled_camera = true;
    return true;
}

bool import_start(::holovibes::gui::MainWindow& mainwindow,
                  Holovibes& holovibes,
                  camera::FrameDescriptor& file_fd,
                  bool& is_enabled_camera,
                  std::string& file_path,
                  unsigned int fps,
                  size_t first_frame,
                  bool load_file_in_gpu,
                  size_t last_frame)
{
    LOG_INFO;

    if (!holovibes.get_cd().is_computation_stopped)
        // if computation is running
        import_stop(mainwindow, holovibes);

    holovibes.get_cd().is_computation_stopped = false;
    // Gather all the usefull data from the ui import panel
    return init_holovibes_import_mode(holovibes,
                                      file_fd,
                                      is_enabled_camera,
                                      file_path,
                                      fps,
                                      first_frame,
                                      load_file_in_gpu,
                                      last_frame);
}

void import_stop(::holovibes::gui::MainWindow& mainwindow, Holovibes& holovibes)
{
    LOG_INFO;

    holovibes.stop_all_worker_controller();
    holovibes.start_information_display(false);

    close_critical_compute(mainwindow, holovibes);

    // FIXME: import_stop() and camera_none() call same methods
    // FIXME: camera_none() weird call because we are dealing with imported file
    mainwindow.camera_none();

    holovibes.get_cd().is_computation_stopped = true;
}

void close_critical_compute(::holovibes::gui::MainWindow& mainwindow, Holovibes& holovibes)
{
    LOG_INFO;
    if (holovibes.get_cd().convolution_enabled)
        mainwindow.set_convolution_mode(false);

    if (holovibes.get_cd().time_transformation_cuts_enabled)
        mainwindow.cancel_time_transformation_cuts();

    holovibes.stop_compute();
}

} // namespace holovibes::api