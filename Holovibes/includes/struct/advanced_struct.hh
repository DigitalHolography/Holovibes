#pragma once

#include "all_struct.hh"

namespace holovibes
{

struct BufferSizes
{
    unsigned input = 512;
    unsigned file = 512;
    unsigned record = 1024;
    unsigned output = 256;
    unsigned time_transformation_cuts = 512;
};

struct Filter2DSmooth
{
    int low = 0;
    int high = 0;
};

struct ContrastThreshold
{
    float lower = 0.5f;
    float upper = 99.5f;
    unsigned cuts_p_offset = 2;
};

struct AdvancedSettings
{
    BufferSizes buffer_size;
    Filter2DSmooth filter2d;
    ContrastThreshold contrast;
    unsigned renorm_constant = 5;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_FWD(BufferSizes)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_FWD(Filter2DSmooth)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_FWD(ContrastThreshold)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_FWD(AdvancedSettings)
} // namespace holovibes