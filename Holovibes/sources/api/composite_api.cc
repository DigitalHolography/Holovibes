#include "composite_api.hh"

namespace holovibes::api
{

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

void set_composite_p_h(int min, int max)
{
    holovibes::CompositeHSV hsv = GET_SETTING(HSV);
    hsv.h.frame_index.min = min;
    hsv.h.frame_index.max = max;
    UPDATE_SETTING(HSV, hsv);
}

void set_composite_intervals(int composite_p_red, int composite_p_blue)
{
    holovibes::CompositeRGB rgb = GET_SETTING(RGB);
    rgb.frame_index.min = composite_p_red;
    rgb.frame_index.max = composite_p_blue;
    UPDATE_SETTING(RGB, rgb);
    pipe_refresh();
}

} // namespace holovibes::api