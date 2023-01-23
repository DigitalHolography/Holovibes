#include "import_pipe_request_on_sync.hh"
#include "API.hh"
#include "input_frame_file.hh"

namespace holovibes
{
template <>
void ImportPipeRequestOnSync::operator()<ImportType>(ImportTypeEnum new_value, Pipe& pipe)
{
    LOG_UPDATE_ON_SYNC(ImportType);

    if (new_value == ImportTypeEnum::None)
    {
        // Shut down all windows
        api::set_raw_view_enabled(false);
        api::set_lens_view_enabled(false);
        api::set_time_transformation_cuts_enable(false);
        api::change_filter2d()->enabled = false;
    }
    // Make sure all windows are in the right state
    pipe.get_view_cache().virtual_synchronize_W<RawViewEnabled>(pipe);
    pipe.get_view_cache().virtual_synchronize_W<LensViewEnabled>(pipe);
    pipe.get_compute_cache().virtual_synchronize_W<TimeTransformationCutsEnable>(pipe);
    pipe.get_compute_cache().virtual_synchronize_W<Filter2D>(pipe);

    if (new_value == ImportTypeEnum::None)
    {
        // Stop the record
        api::detail::change_value<Record>()->is_running = false;
        pipe.get_export_cache().virtual_synchronize_W<Record>(pipe);

        Holovibes::instance().stop_camera_frame_read();
        Holovibes::instance().stop_file_frame_read();

        Holovibes::instance().stop_information_display();
        Holovibes::instance().stop_compute();
    }
    else
    {
       Holovibes::instance().start_information_display();
    }

    // On Camera Import
    if (new_value == ImportTypeEnum::Camera)
    {
        Holovibes::instance().start_camera_frame_read();
    }
    // On File Import
    else if (new_value == ImportTypeEnum::File)
    {
        Holovibes::instance().start_file_frame_read();
    }
}
} // namespace holovibes
