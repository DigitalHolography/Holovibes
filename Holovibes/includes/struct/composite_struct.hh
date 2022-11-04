/*! \file
 *
 * \brief Composite Struct
 *
 */

#pragma once

#include <atomic>

#include "types.hh"
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

/*! \class CompositeRGBStruct
 *
 * \brief Class that represents CompositeRGBStruct
 */
struct CompositeRGBStruct
{
    CompositeP p;
    RGBWeights weight;

    uint get_red() const { return p.min; }
    uint get_blue() const { return p.max; }

    void set_red(uint _red) { p.min = _red; }
    void set_blue(uint _blue) { p.max = _blue; }

    SERIALIZE_JSON_STRUCT(CompositeRGBStruct, p, weight)
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

/*! \class CompositeH
 *
 * \brief Class that represents CompositeH
 */
struct CompositeH
{
    CompositeP p;
    Threshold slider_threshold;
    Threshold threshold;
    Blur blur;

    SERIALIZE_JSON_STRUCT(CompositeH, p, slider_threshold, threshold, blur)
};

/*! \class CompositeSV
 *
 * \brief Class that represents CompositeSV
 */
struct CompositeSV
{
    ActivableCompositeP p;
    Threshold slider_threshold;
    Threshold threshold;

    SERIALIZE_JSON_STRUCT(CompositeSV, p, slider_threshold, threshold)
};

/*! \class CompositeHSVStruct
 *
 * \brief Class that represents CompositeHSVStruct
 */
struct CompositeHSVStruct
{
    CompositeH h{};
    CompositeSV s{};
    CompositeSV v{};

    SERIALIZE_JSON_STRUCT(CompositeHSVStruct, h, s, v)
};

/*! \class Composite
 *
 * \brief Class that represents the composite cache
 */
struct Composite
{
    CompositeKindEnum mode = CompositeKindEnum::RGB;
    bool composite_auto_weights = false;
    CompositeRGBStruct rgb;
    CompositeHSVStruct hsv;

    void Load();
    void Update();

    SERIALIZE_JSON_STRUCT(Composite, mode, composite_auto_weights, rgb, hsv)
};

} // namespace holovibes
