#include "composite_api.hh"

namespace holovibes::api
{

void set_composite_intervals(int composite_p_red, int composite_p_blue)
{
    holovibes::CompositeRGB rgb = GET_SETTING(RGB);
    rgb.frame_index.min = composite_p_red;
    rgb.frame_index.max = composite_p_blue;
    UPDATE_SETTING(RGB, rgb);
    pipe_refresh();
}

void set_composite_intervals_hsv_h_min(uint composite_p_min_h)
{
    set_composite_p_h(composite_p_min_h, get_composite_p_max_h());
    pipe_refresh();
}

void set_composite_intervals_hsv_h_max(uint composite_p_max_h)
{
    set_composite_p_h(get_composite_p_min_h(), composite_p_max_h);
    pipe_refresh();
}

void set_composite_intervals_hsv_s_min(uint composite_p_min_s)
{
    set_composite_p_min_s(composite_p_min_s);
    pipe_refresh();
}

void set_composite_intervals_hsv_s_max(uint composite_p_max_s)
{
    set_composite_p_max_s(composite_p_max_s);
    pipe_refresh();
}

void set_composite_intervals_hsv_v_min(uint composite_p_min_v)
{
    set_composite_p_min_v(composite_p_min_v);
    pipe_refresh();
}

void set_composite_intervals_hsv_v_max(uint composite_p_max_v)
{
    set_composite_p_max_v(composite_p_max_v);
    pipe_refresh();
}

void set_composite_weights(double weight_r, double weight_g, double weight_b)
{
    set_weight_rgb(weight_r, weight_g, weight_b);
    pipe_refresh();
}

void select_composite_rgb() { set_composite_kind(CompositeKind::RGB); }

void select_composite_hsv() { set_composite_kind(CompositeKind::HSV); }

void actualize_frequency_channel_s(bool composite_p_activated_s)
{
    set_composite_p_activated_s(composite_p_activated_s);
}

void actualize_frequency_channel_v(bool composite_p_activated_v)
{
    set_composite_p_activated_v(composite_p_activated_v);
}

} // namespace holovibes::api