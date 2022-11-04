/*! \file
 *
 * \brief Rendering  Struct
 *
 */

#pragma once

#include <vector>

#include "enum_space_transformation.hh"
#include "enum_time_transformation.hh"
#include "enum_computation.hh"
#include "all_struct.hh"

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
    int n1 = 0;
    int n2 = 1;

    SERIALIZE_JSON_STRUCT(Filter2DStruct, enabled, n1, n2)
};

/*! \class ConvolutionStruct
 *
 * \brief Class that represents ConvolutionStruct
 */
struct ConvolutionStruct
{
    bool enabled = false;
    std::string type; // = UID_CONVOLUTION_TYPE_DEFAULT;
    bool divide = false;
    std::vector<float> matrix = {};

    SERIALIZE_JSON_STRUCT(ConvolutionStruct, enabled, type, divide)
};

/*! \class Rendering
 *
 * \brief Class that represents the rendering cache
 */
struct Rendering
{
    Computation image_mode = Computation::Raw;
    unsigned batch_size = 1;
    unsigned time_transformation_stride = 1;
    Filter2DStruct filter2d;
    SpaceTransformationEnum space_transformation = SpaceTransformationEnum::NONE;
    TimeTransformationEnum time_transformation = TimeTransformationEnum::NONE;
    unsigned time_transformation_size = 1;
    float lambda = 852e-9f;
    float z_distance = 1.5f;
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
                          z_distance,
                          convolution)
};

} // namespace holovibes