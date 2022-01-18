#pragma once

#include "all_struct.hh"
#include "enum_composite_kind.hh"

namespace holovibes
{
struct CompositeP
{
    int min = 0;
    int max = 0;

    SERIALIZE_JSON_STRUCT(CompositeP, min, max)
};

struct ActivableCompositeP : public CompositeP
{
    bool activated = false;

    SERIALIZE_JSON_STRUCT(ActivableCompositeP, min, max, activated)
};

struct RGBWeights
{
    float r;
    float g;
    float b;

    SERIALIZE_JSON_STRUCT(RGBWeights, r, g, b)
};

struct CompositeRGB
{
    CompositeP p;
    RGBWeights weight;

    SERIALIZE_JSON_STRUCT(CompositeRGB, p, weight)
};

struct Threshold
{
    float min;
    float max;

    SERIALIZE_JSON_STRUCT(Threshold, min, max)
};

struct Blur
{
    bool enabled = false;
    unsigned kernel_size = 1;

    SERIALIZE_JSON_STRUCT(Blur, enabled, kernel_size)
};

struct CompositeH
{
    CompositeP p;
    Threshold slider_threshold;
    Threshold threshold;
    Blur blur;

    SERIALIZE_JSON_STRUCT(CompositeH, p, slider_threshold, threshold, blur)
};

struct CompositeSV
{
    ActivableCompositeP p;
    Threshold slider_threshold;
    Threshold threshold;

    SERIALIZE_JSON_STRUCT(CompositeSV, p, slider_threshold, threshold)
};

struct CompositeHSV
{
    CompositeH h{};
    CompositeSV s{};
    CompositeSV v{};

    SERIALIZE_JSON_STRUCT(CompositeHSV, h, s, v)
};

struct Composite
{
    CompositeKind mode = CompositeKind::RGB;
    bool composite_auto_weights = false;
    CompositeRGB rgb;
    CompositeHSV hsv;

    SERIALIZE_JSON_STRUCT(Composite, mode, composite_auto_weights, rgb, hsv)
};

} // namespace holovibes
