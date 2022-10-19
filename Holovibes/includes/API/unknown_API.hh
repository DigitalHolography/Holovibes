#pragma once

#include "API_detail.hh"

namespace holovibes::api
{

void check_q_limits();
void init_image_mode(QPoint& position, QSize& size);
bool set_holographic_mode(ushort window_size);
void close_critical_compute();

void open_advanced_settings(QMainWindow* parent, ::holovibes::gui::AdvancedSettingsWindowPanel* specific_panel);

} // namespace holovibes::api
