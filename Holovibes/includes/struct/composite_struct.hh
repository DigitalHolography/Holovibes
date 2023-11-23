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

/*! \class ActivableCompositeP
 *
 * \brief Class that represents ActivableCompositeP
 */
struct ActivableCompositeP : public CompositeP
{
    bool activated = false;

    SERIALIZE_JSON_STRUCT(ActivableCompositeP, min, max, activated)
};

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

/*! \class  Blur
 *
 * \brief Class that represents Blur
 */
struct Blur
{
    bool enabled = false;
    unsigned kernel_size = 1;

    SERIALIZE_JSON_STRUCT(Blur, enabled, kernel_size)
};

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

/*! \class CompositeH
 *
 * \brief Class that represents CompositeH
 */
struct CompositeH : public CompositeChannel
{
    Blur blur;

    SERIALIZE_JSON_STRUCT(CompositeH, frame_index, slider_threshold, threshold, blur)
};

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

    SERIALIZE_JSON_STRUCT(Composite, mode, auto_weight, rgb, hsv)
};

} // namespace holovibes
