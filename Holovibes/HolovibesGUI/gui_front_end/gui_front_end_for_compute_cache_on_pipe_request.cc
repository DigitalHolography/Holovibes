#include "gui_front_end_for_compute_cache_on_pipe_request.hh"
#include "pipe_request_on_sync.hh"
#include "user_interface.hh"
#include "API.hh"

namespace holovibes::gui
{
template <>
void GuiFrontEndForComputeCacheOnPipeRequest::after_method<ComputeMode>()
{
    LOG_UPDATE_FRONT_END_AFTER(ComputeMode);

    if (api::detail::get_value<ImportType>() == ImportTypeEnum::None)
        return;

    const FrameDescriptor& fd = api::get_gpu_input_queue().get_fd();
    unsigned short width = fd.width;
    unsigned short height = fd.height;
    get_good_size(width, height, UserInterface::window_max_size);

    QPoint position(0, 0);
    QSize size(width, height);
    if (UserInterface::instance().xy_window)
    {
        position = UserInterface::instance().xy_window->get_window_position();
        size = UserInterface::instance().xy_window->get_window_size();
    }

    if (api::get_compute_mode() == ComputeModeEnum::Raw)
    {
        UserInterface::instance().main_window->synchronize_thread(
            [=]()
            {
                UserInterface::instance().xy_window.reset(
                    new gui::RawWindow("XY view",
                                       position,
                                       size,
                                       &api::get_gpu_input_queue(),
                                       static_cast<float>(width) / static_cast<float>(height)));
                UserInterface::instance().xy_window->set_is_resize(false);
            },
            true);

        // FIXME INFO - FIXME API
        // std::string fd_info =
        //     std::to_string(fd.width) + "x" + std::to_string(fd.height) + " - " + std::to_string(fd.depth * 8) +
        //     "bit";
    }

    if (api::get_compute_mode() == ComputeModeEnum::Hologram)
    {
        UserInterface::instance().main_window->synchronize_thread(
            [=]()
            {
                UserInterface::instance().xy_window.reset(
                    new gui::HoloWindow("XY view",
                                        position,
                                        size,
                                        &api::get_gpu_output_queue(),
                                        UserInterface::instance().sliceXZ,
                                        UserInterface::instance().sliceYZ,
                                        static_cast<float>(width) / static_cast<float>(height)));
                UserInterface::instance().xy_window->set_is_resize(false);
            },
            true);

        // FIXME INFO - FIXME API
        // std::string fd_info =
        //     std::to_string(fd.width) + "x" + std::to_string(fd.height) + " - " + std::to_string(fd.depth * 8) +
        //     "bit";
    }
}

} // namespace holovibes::gui
