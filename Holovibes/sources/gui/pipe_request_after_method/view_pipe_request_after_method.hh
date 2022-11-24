#pragma once

#include "pipe_request_on_sync.hh"
#include "API.hh"
#include "user_interface.hh"

namespace holovibes::gui
{

class GuiSetterViewPipeRequestAfterMethod
{
  public:
    static void set_after_method_of_pipe_request() {}

  private:
    void chart_display_change_()
    {
        if (api::detail::get_value<ChartDisplayEnabled>())
        {
            UserInterface::instance().plot_window_ =
                std::make_unique<gui::PlotWindow>(*api::get_compute_pipe().get_chart_display_queue_ptr(),
                                                  UserInterface::instance().auto_scale_point_threshold_,
                                                  "Chart");

            UserInterface::instance().get_export_panel()->connect(UserInterface::instance().plot_window_.get(),
                                                                  SIGNAL(closed()),
                                                                  this,
                                                                  SLOT(stop_chart_display()),
                                                                  Qt::UniqueConnection);

            UserInterface::instance().get_export_panel()->ChartPlotPushButton->setEnabled(false);
        }
        else
        {
            UserInterface::instance().plot_window_.reset(nullptr);
            UserInterface::instance().get_export_panel()->ChartPlotPushButton->setEnabled(true);
        }
    }
};

} // namespace holovibes::gui