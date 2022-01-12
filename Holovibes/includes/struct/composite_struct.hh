#pragma once

#include "all_struct.hh"
#include "enum_composite_kind.hh"

namespace holovibes
{
struct CompositeP
{
    int min = 0;
    int max = 0;
};

struct ActivableCompositeP : public CompositeP
{
    bool activated = false;
};

struct RGBWeights
{
    float r;
    float g;
    float b;
};

struct CompositeRGB
{
    CompositeP p;
    RGBWeights weight;
};

struct Threshold
{
    float min;
    float max;
};

struct Blur
{
    bool enabled = false;
    unsigned kernel_size = 1;
};

struct CompositeH
{
    CompositeP p;
    Threshold slider_threshold;
    Threshold threshold;
    Blur blur;
};

struct CompositeSV
{
    ActivableCompositeP p;
    Threshold slider_threshold;
    Threshold threshold;
};

struct CompositeHSV
{
    CompositeH h{};
    CompositeSV s{};
    CompositeSV v{};
};

struct Composite
{
    CompositeKind mode = CompositeKind::RGB;
    bool composite_auto_weights = false;
    CompositeRGB rgb;
    CompositeHSV hsv;
};

SERIALIZE_JSON_FWD(CompositeP)
SERIALIZE_JSON_FWD(ActivableCompositeP)
SERIALIZE_JSON_FWD(RGBWeights)
SERIALIZE_JSON_FWD(CompositeRGB)
SERIALIZE_JSON_FWD(Threshold)
SERIALIZE_JSON_FWD(Blur)
SERIALIZE_JSON_FWD(CompositeH)
SERIALIZE_JSON_FWD(CompositeSV)
SERIALIZE_JSON_FWD(CompositeHSV)
SERIALIZE_JSON_FWD(Composite)

} // namespace holovibes
