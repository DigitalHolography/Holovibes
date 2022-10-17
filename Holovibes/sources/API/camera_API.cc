#include "API.hh"

namespace holovibes::api
{

void camera_none()
{
    api::close_windows();
    api::close_critical_compute();

    if (get_compute_mode() == Computation::Hologram)

        Holovibes::instance().stop_compute();
    Holovibes::instance().stop_frame_read();

    UserInterfaceDescriptor::instance().is_enabled_camera_ = false;
    set_is_computation_stopped(true);

    UserInterfaceDescriptor::instance().import_type_ = ImportType::None;
}

bool change_camera(CameraKind c)
{
    camera_none();

    if (c == CameraKind::NONE)
        return false;

    try
    {
        UserInterfaceDescriptor::instance().mainDisplay.reset(nullptr);
        if (get_compute_mode() == Computation::Raw)
            Holovibes::instance().stop_compute();
        Holovibes::instance().stop_frame_read();

        Holovibes::instance().start_camera_frame_read(c);
        UserInterfaceDescriptor::instance().is_enabled_camera_ = true;
        UserInterfaceDescriptor::instance().kCamera = c;

        set_is_computation_stopped(false);

        return true;
    }
    catch (const camera::CameraException& e)
    {
        LOG_ERROR(main, "[CAMERA] {}", e.what());
    }
    catch (const std::exception& e)
    {
        LOG_ERROR(main, "Catch {}", e.what());
    }

    return false;
}

void configure_camera()
{
    QDesktopServices::openUrl(QUrl::fromLocalFile(QString::fromStdString(Holovibes::instance().get_camera_ini_name())));
}

} // namespace holovibes::api
