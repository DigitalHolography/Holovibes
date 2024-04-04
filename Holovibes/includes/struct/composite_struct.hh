/*! \file
 *
 * \brief Composite Struct
 *
 */

#pragma once
#include "all_struct.hh"
#include "enum_composite_kind.hh"

namespace holovibes
{

/*! \class CompositeP
 *
 * \brief Class that represents CompositeP
 */
struct CompositeP
{
    int min = 0;
    int max = 0;

    SERIALIZE_JSON_STRUCT(CompositeP, min, max)
};

inline bool operator==(const CompositeP& lhs, const CompositeP& rhs)
{
    return lhs.min == rhs.min && lhs.max == rhs.max;
}

/*! \class ActivableCompositeP
 *
 * \brief Class that represents ActivableCompositeP
 */
struct ActivableCompositeP : public CompositeP
{
    bool activated = false;

    SERIALIZE_JSON_STRUCT(ActivableCompositeP, min, max, activated)
};

inline bool operator==(const ActivableCompositeP& lhs, const ActivableCompositeP& rhs)
{
    return lhs.min == rhs.min && lhs.max == rhs.max && lhs.activated == rhs.activated;
}

/*! \class RGBWeights
 *
 * \brief Class that represents RGBWeights
 */
struct RGBWeights
{
    float r;
    float g;
    float b;

    SERIALIZE_JSON_STRUCT(RGBWeights, r, g, b)
};

inline bool operator==(const RGBWeights& lhs, const RGBWeights& rhs)
{
    return lhs.r == rhs.r && lhs.g == rhs.g && lhs.b == rhs.b;
}

/*! \class CompositeRGB
 *
 * \brief Class that represents CompositeRGB
 */
struct CompositeRGB
{
    CompositeP frame_index;
    RGBWeights weight;

    SERIALIZE_JSON_STRUCT(CompositeRGB, frame_index, weight)
};

inline bool operator==(const CompositeRGB& lhs, const CompositeRGB& rhs)
{
    return lhs.frame_index == rhs.frame_index && lhs.weight == rhs.weight;
}

/*! \class Threshold
 *
 * \brief Class that represents Threshold
 */
struct Threshold
{
    float min;
    float max;

    SERIALIZE_JSON_STRUCT(Threshold, min, max)
};

inline bool operator==(const Threshold& lhs, const Threshold& rhs)
{
    return lhs.min == rhs.min && lhs.max == rhs.max;
}

/*! \class CompositeChannel
 *
 * \brief Class that represents CompositeChannel
 */
struct CompositeChannel
{
    ActivableCompositeP frame_index;
    Threshold slider_threshold = {0.0f, 1.0f};
    Threshold threshold = {0.001f, 99.8f};

    SERIALIZE_JSON_STRUCT(CompositeChannel, frame_index, slider_threshold, threshold)
};

inline bool operator==(const CompositeChannel& lhs, const CompositeChannel& rhs)
{
    return lhs.frame_index == rhs.frame_index && lhs.slider_threshold == rhs.slider_threshold && lhs.threshold == rhs.threshold;
}

/*! \class CompositeH
 *
 * \brief Class that represents CompositeH
 */
struct CompositeH : public CompositeChannel
{
    Threshold slider_shift = {0.0f, 1.0f};

    SERIALIZE_JSON_STRUCT(CompositeH, frame_index, slider_threshold, threshold, slider_shift)
};

inline bool operator==(const CompositeH& lhs, const CompositeH& rhs)
{
    return lhs.frame_index == rhs.frame_index && lhs.slider_threshold == rhs.slider_threshold && lhs.threshold == rhs.threshold;
}

/*! \class CompositeHSV
 *
 * \brief Class that represents CompositeHSV
 */
struct CompositeHSV
{
    CompositeH h{};
    CompositeChannel s{};
    CompositeChannel v{};

    SERIALIZE_JSON_STRUCT(CompositeHSV, h, s, v)
};

inline bool operator==(const CompositeHSV& lhs, const CompositeHSV& rhs)
{
    return lhs.h == rhs.h && lhs.s == rhs.s && lhs.v == rhs.v;
}

/*! \class Composite
 *
 * \brief Class that represents the composite cache
 */
struct Composite
{
    CompositeKind mode = CompositeKind::RGB;
    bool auto_weight = false;
    CompositeRGB rgb;
    CompositeHSV hsv;

    void Load();
    void Update();
    void Assert(bool cli) const;

    SERIALIZE_JSON_STRUCT(Composite, mode, auto_weight, rgb, hsv)
};

} // namespace holovibes
