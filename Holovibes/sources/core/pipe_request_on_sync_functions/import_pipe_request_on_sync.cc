#include "import_pipe_request_on_sync.hh"
#include "API.hh"
#include "input_frame_file.hh"

namespace holovibes
{
template <>
void ImportPipeRequestOnSync::operator()<ImportType>(ImportTypeEnum new_value, Pipe& pipe)
{
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
        Holovibes::instance().stop_and_join_camera_frame_read();
        Holovibes::instance().stop_and_join_file_frame_read();

        Holovibes::instance().stop_and_join_frame_record();
        Holovibes::instance().stop_and_join_chart_record();

        Holovibes::instance().stop_and_join_information_display();
        Holovibes::instance().stop_compute();

        // We are currently stopping all worker, so the pipe does not
        // need to refresh
        disable_pipe();
        return;
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
