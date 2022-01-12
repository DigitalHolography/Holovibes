#pragma once

#include "enum_space_transformation.hh"
#include "enum_time_transformation.hh"
#include "enum_computation.hh"
#include "all_struct.hh";

namespace holovibes
{
struct Filter2D
{

    bool enabled = false;
    int n1 = 0;
    int n1 = 1;
};

struct Convolution
{
    bool enabled = false;
    std::string type;
    bool divide = false;
};

struct Rendering
{
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
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_FWD(Filter2D)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_FWD(Convolution)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_FWD(Rendering)

} // namespace holovibes