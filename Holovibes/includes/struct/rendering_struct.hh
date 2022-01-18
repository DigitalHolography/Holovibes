#pragma once

#include "enum_space_transformation.hh"
#include "enum_time_transformation.hh"
#include "enum_computation.hh"
#include "all_struct.hh"

namespace holovibes
{

struct Rendering
{

    struct Filter2D
    {
        bool enabled = false;
        int n1 = 0;
        int n2 = 1;

        SERIALIZE_JSON_STRUCT(Filter2D, enabled, n1, n2)
    };

    struct Convolution
    {
        bool enabled = false;
        std::string type;
        bool divide = false;

        SERIALIZE_JSON_STRUCT(Convolution, enabled, type, divide)
    };

    Computation image_mode = Computation::Raw;
    unsigned batch_size = 1;
    unsigned time_transformation_stride = 1;
    Filter2D filter2d;
    SpaceTransformation space_transformation = SpaceTransformation::NONE;
    TimeTransformation time_transformation = TimeTransformation::NONE;
    unsigned time_transformation_size = 1;
    float lambda = 852e-9f;
    float z_distance = 1.5f;
    Convolution convolution;

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