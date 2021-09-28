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

bool init_holovibesimport_mode(Holovibes& holovibes,
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

bool import_start(Holovibes& holovibes,
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
        import_stop(is_enabled_camera, holovibes);

    holovibes.get_cd().is_computation_stopped = false;
    // Gather all the usefull data from the ui import panel
    return init_holovibesimport_mode(holovibes,
                                     file_fd,
                                     is_enabled_camera,
                                     file_path,
                                     fps,
                                     first_frame,
                                     load_file_in_gpu,
                                     last_frame);
}

void import_stop(bool& is_enabled_camera, Holovibes& holovibes)
{
    LOG_INFO;

    holovibes.stop_all_worker_controller();
    holovibes.start_information_display(false);

    close_critical_compute(holovibes);

    // FIXME: import_stop() and camera_none() call same methods
    // FIXME: camera_none() weird call because we are dealing with imported file
    camera_none(is_enabled_camera, holovibes);

    holovibes.get_cd().is_computation_stopped = true;
}

void camera_none(bool& is_enabled_camera, Holovibes& holovibes)
{
    LOG_INFO;
    close_critical_compute(holovibes);
    if (!is_raw_mode(holovibes))
        holovibes.stop_compute();
    holovibes.stop_frame_read();
    remove_infos();

    is_enabled_camera = false;
    holovibes.get_cd().is_computation_stopped = true;
}

void close_critical_compute(Holovibes& holovibes)
{
    LOG_INFO;
    if (holovibes.get_cd().convolution_enabled)
        set_convolution_mode(holovibes, false);

    if (holovibes.get_cd().time_transformation_cuts_enabled)
        cancel_time_transformation_cuts(holovibes, []() { return; });

    holovibes.stop_compute();
}

bool is_raw_mode(Holovibes& holovibes)
{
    LOG_INFO;
    return holovibes.get_cd().compute_mode == Computation::Raw;
}

void remove_infos()
{
    LOG_INFO;
    Holovibes::instance().get_info_container().clear();
}

void set_convolution_mode(Holovibes& holovibes, const bool value)
{
    LOG_INFO;

    try
    {
        auto pipe = holovibes.get_compute_pipe();

        if (value)
        {
            pipe->request_convolution();
            // Wait for the convolution to be enabled for notify
            while (pipe->get_convolution_requested())
                continue;
        }
        else
        {
            pipe->request_disable_convolution();
            // Wait for the convolution to be disabled for notify
            while (pipe->get_disable_convolution_requested())
                continue;
        }
    }
    catch (const std::exception& e)
    {
        holovibes.get_cd().convolution_enabled = false;
        LOG_ERROR << e.what();
    }
}

void cancel_time_transformation_cuts(Holovibes& holovibes, std::function<void()> callback)
{
    LOG_INFO;
    if (holovibes.get_cd().time_transformation_cuts_enabled)
    {

        holovibes.get_cd().contrast_max_slice_xz = false;
        holovibes.get_cd().contrast_max_slice_yz = false;
        holovibes.get_cd().log_scale_slice_xz_enabled = false;
        holovibes.get_cd().log_scale_slice_yz_enabled = false;
        holovibes.get_cd().img_acc_slice_xz_enabled = false;
        holovibes.get_cd().img_acc_slice_yz_enabled = false;

        holovibes.get_compute_pipe().get()->insert_fn_end_vect(callback);

        try
        {
            // Wait for refresh to be enabled for notify
            while (holovibes.get_compute_pipe()->get_refresh_request())
                continue;
        }
        catch (const std::exception& e)
        {
            LOG_ERROR << e.what();
        }

        holovibes.get_cd().time_transformation_cuts_enabled = false;
    }
}

} // namespace holovibes::api