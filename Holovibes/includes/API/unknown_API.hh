#pragma once

#include "API_detail.hh"

namespace holovibes::api
{

void check_p_limits();
void check_q_limits();
void init_image_mode(QPoint& position, QSize& size);
void set_holographic_mode();
void close_critical_compute();

bool slide_update_threshold(
    const int slider_value, float& receiver, float& bound_to_update, const float lower_bound, const float upper_bound);

void set_log_scale(const bool value);
bool slide_update_threshold(
    const int slider_value, float& receiver, float& bound_to_update, const float lower_bound, const float upper_bound);

} // namespace holovibes::api
