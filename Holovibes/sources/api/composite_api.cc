#include "composite_api.hh"

namespace holovibes::api
{

#pragma region RGB

void set_weight_rgb(double r, double g, double b)
{
    holovibes::CompositeRGB rgb = GET_SETTING(RGB);
    rgb.weight.r = r;
    rgb.weight.g = g;
    rgb.weight.b = b;
    UPDATE_SETTING(RGB, rgb);
    pipe_refresh();
}

void set_rgb_p(int min, int max)
{
    holovibes::CompositeRGB rgb = GET_SETTING(RGB);
    rgb.frame_index.min = min;
    rgb.frame_index.max = max;
    UPDATE_SETTING(RGB, rgb);
}

void set_composite_auto_weights(bool value)
{
    UPDATE_SETTING(CompositeAutoWeights, value);
    pipe_refresh();
}

void set_composite_intervals(int composite_p_red, int composite_p_blue)
{
    holovibes::CompositeRGB rgb = GET_SETTING(RGB);
    rgb.frame_index.min = composite_p_red;
    rgb.frame_index.max = composite_p_blue;
    UPDATE_SETTING(RGB, rgb);
    pipe_refresh();
}

#pragma endregion

#pragma region HSV Hue

void set_composite_p_min_h(uint value)
{
    SET_SETTING(HSV, h.frame_index.min, value);
    pipe_refresh();
}

void set_composite_p_max_h(uint value)
{
    SET_SETTING(HSV, h.frame_index.max, value);
    pipe_refresh();
}

#pragma endregion

#pragma region HSV Saturation

void set_composite_p_min_s(uint value)
{
    SET_SETTING(HSV, s.frame_index.min, value);
    if (get_composite_p_activated_s())
        pipe_refresh();
}

void set_composite_p_max_s(uint value)
{
    SET_SETTING(HSV, s.frame_index.max, value);
    if (get_composite_p_activated_s())
        pipe_refresh();
}

#pragma endregion

#pragma region HSV Value

void set_composite_p_min_v(uint value)
{
    SET_SETTING(HSV, v.frame_index.min, value);
    if (get_composite_p_activated_v())
        pipe_refresh();
}

void set_composite_p_max_v(uint value)
{
    SET_SETTING(HSV, v.frame_index.max, value);
    if (get_composite_p_activated_v())
        pipe_refresh();
}

#pragma endregion

void set_composite_p_h(int min, int max)
{
    holovibes::CompositeHSV hsv = GET_SETTING(HSV);
    hsv.h.frame_index.min = min;
    hsv.h.frame_index.max = max;
    UPDATE_SETTING(HSV, hsv);
}

} // namespace holovibes::api