#pragma once

#include "API_detail.hh"

namespace holovibes::api
{

const View_Window& get_current_window() { return api::get_current_window(); }
void change_window(const int index) { api::change_window(index); }
std::unique_ptr<::holovibes::gui::RawWindow>& get_main_display();
std::unique_ptr<::holovibes::gui::RawWindow>& get_raw_window();
void start_chart_display();
void stop_chart_display();
double get_rotation();
bool get_flip_enabled();
unsigned GSH::get_img_accu_level();

} // namespace holovibes::api
