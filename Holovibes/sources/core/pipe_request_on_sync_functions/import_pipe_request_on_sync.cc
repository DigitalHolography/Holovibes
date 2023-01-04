#include "import_pipe_request_on_sync.hh"
#include "API.hh"
#include "input_frame_file.hh"

namespace holovibes
{

template <>
void ImportPipeRequestOnSync::on_sync<ImportType>(ImportTypeEnum new_value, ImportTypeEnum old_value, Pipe& pipe)
{
    if (old_value == ImportTypeEnum::None)
    {
        operator()<ImportType>(new_value, pipe);
    }

    // FIXME API : miss case when reload
}

template <>
void ImportPipeRequestOnSync::operator()<ImportType>(ImportTypeEnum new_value, Pipe& pipe)
{
    pipe.get_view_cache().virtual_synchronize_W<RawViewEnabled>(pipe);
    pipe.get_view_cache().virtual_synchronize_W<LensViewEnabled>(pipe);
    pipe.get_compute_cache().virtual_synchronize_W<TimeTransformationCutsEnable>(pipe);

    if (new_value == ImportTypeEnum::None)
    {
        Holovibes::instance().stop_camera_frame_read();
        Holovibes::instance().stop_file_frame_read();

        Holovibes::instance().stop_information_display();
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

template <>
void ImportPipeRequestOnSync::on_sync<CurrentCameraKind>(CameraKind new_value, CameraKind old_value, Pipe& pipe)
{
    operator()<CurrentCameraKind>(new_value, pipe);
}

template <>
void ImportPipeRequestOnSync::operator()<CurrentCameraKind>(CameraKind new_value, Pipe& pipe)
{
    // Stop Camera
    if (new_value == CameraKind::None)
    {
    }
    // Start Camera
    else
    {
    }
}
} // namespace holovibes