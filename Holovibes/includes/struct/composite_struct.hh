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
}

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
}

struct Blur
{
    bool enabled = false;
    unsigned kernel_size = 1;
}

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

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_FWD(CompositeP)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_FWD(ActivableCompositeP)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_FWD(RGBWeights)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_FWD(CompositeRGB)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_FWD(Threshold)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_FWD(Blur)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_FWD(CompositeH)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_FWD(CompositeSV)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_FWD(CompositeHSV)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_FWD(Composite)

} // namespace holovibes
