#pragma once

namespace holovibes::entities
{
struct BatchQuery
{
    const unsigned int batch_size;
};

struct BatchCommand
{
    const unsigned int batch_size;
};

struct TimeTranformationStrideQuery
{
    const unsigned int time_transformation_stride;
};

struct TimeTranformationStrideCommand
{
    const unsigned int time_transformation_stride;
};

struct Span
{
    const float min;
    const float max;
};

struct HSVSpan
{
    const Span h;
    const Span s;
    const Span v;
};

struct RGB
{
    const Span p;
    const float r;
    const float g;
    const float b;
};

struct NotifyCompositePanel
{
    const bool composite_auto_weights;

    const RGB rgb;

    // HSV
    const HSVSpan slider_hsv_span;  // previously slider_threshold_min and max
    const HSVSpan hsv_span;         // previously threshold_min and max

    // H
    const bool h_blur_enabled;
    const unsigned int h_blur_kernel_size;

    // S
    const bool s_p_enabled;

    // V
    const bool v_p_enabled;
};

} // namespace holovibes::entities
