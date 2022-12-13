#include "gui_front_end_for_import_cache_on_pipe_request.hh"
#include "pipe_request_on_sync.hh"
#include "user_interface.hh"
#include "API.hh"

#include "gui_front_end_for_compute_cache_on_pipe_request.hh"

namespace holovibes::gui
{

template <>
void GuiFrontEndForImportCacheOnPipeRequest::before_method<ImportType>()
{
    LOG_UPDATE_FRONT_END_BEFORE(ImportType);

    if (api::detail::get_value<ImportType>() == ImportTypeEnum::None)
    {
        UserInterface::instance().main_window->synchronize_thread(
            [=]()
            {
                UserInterface::instance().xy_window.reset(nullptr);
                UserInterface::instance().sliceXZ.reset(nullptr);
                UserInterface::instance().sliceYZ.reset(nullptr);
                UserInterface::instance().filter2d_window.reset(nullptr);

                if (UserInterface::instance().lens_window)
                    api::set_lens_view_enabled(false);
                if (UserInterface::instance().raw_window)
                    api::set_raw_view_enabled(false);

                UserInterface::instance().plot_window_.reset(nullptr);
            }, true);
    }
}

template <>
void GuiFrontEndForImportCacheOnPipeRequest::after_method<ImportType>()
{
    LOG_UPDATE_FRONT_END_AFTER(ImportType);

    if (api::detail::get_value<ImportType>() != ImportTypeEnum::None)
    {
        UserInterface::instance().main_window->layout_toggled();
        GuiFrontEndForComputeCacheOnPipeRequest::after_method<ComputeMode>();
    }
}

} // namespace holovibes::gui
