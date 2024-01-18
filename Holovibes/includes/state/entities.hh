#pragma once

namespace holovibes::entities
{
struct BatchQuery
{
    unsigned int value;
};

struct BatchCommand
{
    unsigned int value;
};

struct TimeStrideQuery
{
    unsigned int value;
};

struct TimeStrideCommand
{
    unsigned int value;
};

struct TimeTransformationSizeQuery
{
    unsigned int value;
};

struct TimeTransformationSizeCommand
{
    unsigned int value;
};

template <typename T = int>
struct Span
{
    const T min;
    const T max;
};

struct HSVSpan
{
    const Span<unsigned int> h;
    const Span<unsigned int> s;
    const Span<unsigned int> v;
};

struct RGB
{
    const Span<int> p;
    const float r;
    const float g;
    const float b;
};

struct NotifyCompositePanel
{
    const bool composite_auto_weights;

    const RGB rgb;

    // HSV
    const HSVSpan slider_hsv_span; // previously slider_threshold_min and max
    const HSVSpan hsv_span;        // previously threshold_min and max

    // S
    const bool s_p_enabled;

    // V
    const bool v_p_enabled;
};

} // namespace holovibes::entities
