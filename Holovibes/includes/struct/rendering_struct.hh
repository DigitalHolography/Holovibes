/*! \file
 *
 * \brief Rendering  Struct
 *
 */

#pragma once

#include <vector>

#include "enum_space_transformation.hh"
#include "enum_time_transformation.hh"
#include "enum_compute_mode.hh"
#include "json_macro.hh"

#define UID_CONVOLUTION_TYPE_DEFAULT "None"

namespace holovibes
{

/*! \class Filter2DStruct
 *
 * \brief Class that represents Filter2DStruct
 */
struct Filter2DStruct
{
    bool enabled = false;
    int inner_radius = 0;
    int outer_radius = 1;

    SERIALIZE_JSON_STRUCT(Filter2DStruct, enabled, inner_radius, outer_radius)

    bool operator!=(const Filter2DStruct& rhs) const
    {
        return enabled != rhs.enabled || inner_radius != rhs.inner_radius || outer_radius != rhs.outer_radius;
    }
};

/*! \class ConvolutionStruct
 *
 * \brief Class that represents ConvolutionStruct
 */
struct ConvolutionStruct
{
    std::string type = UID_CONVOLUTION_TYPE_DEFAULT;
    bool divide = false;

    SERIALIZE_JSON_STRUCT(ConvolutionStruct, type, divide)

    bool operator!=(const ConvolutionStruct& rhs) const { return divide != rhs.divide || type != rhs.type; }

    bool is_enabled() const { return type != UID_CONVOLUTION_TYPE_DEFAULT; }
    void disable() { type = UID_CONVOLUTION_TYPE_DEFAULT; }
};

/*! \class Rendering
 *
 * \brief Class that represents the rendering cache
 */
struct Rendering
{
    ComputeModeEnum image_mode = ComputeModeEnum::Raw;
    unsigned batch_size = 1;
    unsigned time_transformation_stride = 1;
    Filter2DStruct filter2d;
    SpaceTransformationEnum space_transformation = SpaceTransformationEnum::NONE;
    TimeTransformationEnum time_transformation = TimeTransformationEnum::NONE;
    unsigned time_transformation_size = 1;
    float lambda = 852e-9f;
    float propagation_distance = 1.5f;
    ConvolutionStruct convolution;

    void Update();
    void Load();

    SERIALIZE_JSON_STRUCT(Rendering,
                          image_mode,
                          batch_size,
                          time_transformation_stride,
                          filter2d,
                          space_transformation,
                          time_transformation,
                          time_transformation_size,
                          lambda,
                          propagation_distance,
                          convolution)
};

inline std::ostream& operator<<(std::ostream& os, const Filter2DStruct& value) { return os << json{value}; }
inline std::ostream& operator<<(std::ostream& os, const ConvolutionStruct& value) { return os << json{value}; }
inline std::ostream& operator<<(std::ostream& os, const Rendering& value) { return os << json{value}; }

} // namespace holovibes
