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

    bool operator!=(const CompositeP& rhs)const { return min != rhs.min || max != rhs.max; }
};

/*! \class ActivableCompositeP
 *
 * \brief Class that represents ActivableCompositeP
 */
struct ActivableCompositeP : public CompositeP
{
    bool activated = false;

    SERIALIZE_JSON_STRUCT(ActivableCompositeP, min, max, activated)

    bool operator!=(const ActivableCompositeP& rhs)const { return activated != rhs.activated; }
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

    bool operator!=(const RGBWeights& rhs)const { return r != rhs.r || g != rhs.g || b != rhs.b; }
};

/*! \class CompositeRGBStruct
 *
 * \brief Class that represents CompositeRGBStruct
 */
struct CompositeRGBStruct
{
    CompositeP frame_index;
    RGBWeights weight;

    SERIALIZE_JSON_STRUCT(CompositeRGBStruct, frame_index, weight)

    bool operator!=(const CompositeRGBStruct& rhs) const{ return frame_index != rhs.frame_index || weight != rhs.weight; }
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

    bool operator!=(const Threshold& rhs) const{ return min != rhs.min || max != rhs.max; }
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

    bool operator!=(const Blur& rhs) const{ return enabled != rhs.enabled || kernel_size != rhs.kernel_size; }
};

/*! \class CompositeH
 *
 * \brief Class that represents CompositeH
 */
struct CompositeH
{
    CompositeP frame_index;
    Threshold slider_threshold;
    Threshold threshold;
    Blur blur;

    SERIALIZE_JSON_STRUCT(CompositeH, frame_index, slider_threshold, threshold, blur)

    bool operator!=(const CompositeH& rhs)const
    {
        return frame_index != rhs.frame_index || slider_threshold != rhs.slider_threshold || threshold != rhs.threshold || blur != rhs.blur;
    }
};

/*! \class CompositeSV
 *
 * \brief Class that represents CompositeSV
 */
struct CompositeSV
{
    ActivableCompositeP frame_index;
    Threshold slider_threshold;
    Threshold threshold;

    SERIALIZE_JSON_STRUCT(CompositeSV, frame_index, slider_threshold, threshold)

    bool operator!=(const CompositeSV& rhs)const
    {
        return frame_index != rhs.frame_index || slider_threshold != rhs.slider_threshold || threshold != rhs.threshold;
    }
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

    bool operator!=(const CompositeHSVStruct& rhs) const{ return h != rhs.h || s != rhs.s || v != rhs.v; }
};

/*! \class Composite
 *
 * \brief Class that represents the composite cache
 */
struct Composite
{
    CompositeKindEnum mode = CompositeKindEnum::RGB;
    bool auto_weight = false;
    CompositeRGBStruct rgb;
    CompositeHSVStruct hsv;

    void Load();
    void Update();

    SERIALIZE_JSON_STRUCT(Composite, mode, auto_weight, rgb, hsv)
};

inline std::ostream& operator<<(std::ostream& os, const CompositeP& value) { return os << json{value}; }
inline std::ostream& operator<<(std::ostream& os, const ActivableCompositeP& value) { return os << json{value}; }
inline std::ostream& operator<<(std::ostream& os, const RGBWeights& value) { return os << json{value}; }
inline std::ostream& operator<<(std::ostream& os, const Threshold& value) { return os << json{value}; }
inline std::ostream& operator<<(std::ostream& os, const Blur& value) { return os << json{value}; }
inline std::ostream& operator<<(std::ostream& os, const CompositeH& value) { return os << json{value}; }
inline std::ostream& operator<<(std::ostream& os, const CompositeSV& value) { return os << json{value}; }
inline std::ostream& operator<<(std::ostream& os, const Composite& value) { return os << json{value}; }

inline std::ostream& operator<<(std::ostream& os, const CompositeHSVStruct& value) { return os /*<< json{value}*/; }
inline std::ostream& operator<<(std::ostream& os, const CompositeRGBStruct& value) { return os /*<< json{value}*/; }
} // namespace holovibes
