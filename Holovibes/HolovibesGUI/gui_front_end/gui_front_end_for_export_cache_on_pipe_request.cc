#include "gui_front_end_for_export_cache_on_pipe_request.hh"
#include "pipe_request_on_sync.hh"
#include "user_interface.hh"
#include "API.hh"

namespace holovibes::gui
{

template <>
void GuiFrontEndForExportCacheOnPipeRequest::after_method<Record>()
{
    LOG_UPDATE_FRONT_END_BEFORE(Record);

    if (api::get_record().is_running == false)
        return;
    if (UserInterface::instance().xy_window == nullptr)
        return;

    if (api::get_record().record_type == RecordStruct::RecordType::CHART)
    {
        UserInterface::instance().main_window->synchronize_thread(
            [=]()
            {
                UserInterface::instance().xy_window->resetTransform();
                UserInterface::instance().xy_window->getOverlayManager().enable_all(KindOfOverlay::Signal);
                UserInterface::instance().xy_window->getOverlayManager().enable_all(KindOfOverlay::Noise);
                UserInterface::instance().xy_window->getOverlayManager().create_overlay<KindOfOverlay::Signal>();
            });
    }
    else
    {
        UserInterface::instance().main_window->synchronize_thread(
            [=]()
            {
                UserInterface::instance().xy_window->resetTransform();
                UserInterface::instance().xy_window->getOverlayManager().disable_all(KindOfOverlay::Signal);
                UserInterface::instance().xy_window->getOverlayManager().disable_all(KindOfOverlay::Noise);
            });
    }
}

} // namespace holovibes::gui
