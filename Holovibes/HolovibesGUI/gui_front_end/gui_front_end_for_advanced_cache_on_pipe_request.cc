#include "pipe_request_on_sync.hh"
#include "user_interface.hh"
#include "API.hh"

#include "gui_front_end_for_advanced_cache_on_pipe_request.hh"

namespace holovibes::gui
{

template <>
void GuiFrontEndForAdvancedCacheOnPipeRequest::after_method<RawBitshift>()
{
    LOG_UPDATE_FRONT_END_BEFORE(RawBitshift);

    if (UserInterface::instance().xy_window)
        UserInterface::instance().xy_window->update_bitshift();
}

} // namespace holovibes::gui
