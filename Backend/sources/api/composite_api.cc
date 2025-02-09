#include "composite_api.hh"

#include "API.hh"

namespace holovibes::api
{

#pragma region RGB

void CompositeApi::set_weight_rgb(float r, float g, float b) const
{
    holovibes::CompositeRGB rgb = GET_SETTING(RGB);
    rgb.weight.r = r;
    rgb.weight.g = g;
    rgb.weight.b = b;
    UPDATE_SETTING(RGB, rgb);
}

void CompositeApi::set_rgb_p(int min, int max) const
{
    holovibes::CompositeRGB rgb = GET_SETTING(RGB);
    rgb.frame_index.min = min;
    rgb.frame_index.max = max;
    UPDATE_SETTING(RGB, rgb);
}

void CompositeApi::set_composite_auto_weights(bool value) const { UPDATE_SETTING(CompositeAutoWeights, value); }

#pragma endregion

#pragma region HSV Hue

void CompositeApi::set_composite_p_min_h(uint value) const { SET_SETTING(HSV, h.frame_index.min, value); }

void CompositeApi::set_composite_p_max_h(uint value) const { SET_SETTING(HSV, h.frame_index.max, value); }

#pragma endregion

#pragma region HSV Saturation

void CompositeApi::set_composite_p_min_s(uint value) const
{
    if (!get_composite_p_activated_s())
        return;

    SET_SETTING(HSV, s.frame_index.min, value);
}

void CompositeApi::set_composite_p_max_s(uint value) const
{
    if (!get_composite_p_activated_s())
        return;

    SET_SETTING(HSV, s.frame_index.max, value);
}

#pragma endregion

#pragma region HSV Value

void CompositeApi::set_composite_p_min_v(uint value) const
{
    if (!get_composite_p_activated_v())
        return;

    SET_SETTING(HSV, v.frame_index.min, value);
}

void CompositeApi::set_composite_p_max_v(uint value) const
{
    if (!get_composite_p_activated_v())
        return;

    SET_SETTING(HSV, v.frame_index.max, value);
}

#pragma endregion

void CompositeApi::set_composite_p_h(int min, int max) const
{
    holovibes::CompositeHSV hsv = GET_SETTING(HSV);
    hsv.h.frame_index.min = min;
    hsv.h.frame_index.max = max;
    UPDATE_SETTING(HSV, hsv);
}

} // namespace holovibes::api