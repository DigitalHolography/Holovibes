#pragma once

#include "all_struct.hh"

namespace holovibes
{

struct AdvancedSettings
{
    struct BufferSizes
    {
        unsigned input = 512;
        unsigned file = 512;
        unsigned record = 1024;
        unsigned output = 256;
        unsigned time_transformation_cuts = 512;

        void Update();

        SERIALIZE_JSON_STRUCT(BufferSizes, input, file, record, output, time_transformation_cuts)
    };

    struct Filter2DSmooth
    {
        int low = 0;
        int high = 0;

        void Update();

        SERIALIZE_JSON_STRUCT(Filter2DSmooth, low, high)
    };

    struct ContrastThreshold
    {
        float lower = 0.5f;
        float upper = 99.5f;
        unsigned cuts_p_offset = 2;

        void Update();

        SERIALIZE_JSON_STRUCT(ContrastThreshold, lower, upper, cuts_p_offset)
    };

    BufferSizes buffer_size;
    Filter2DSmooth filter2d_smooth;
    ContrastThreshold contrast;
    unsigned renorm_constant = 5;

    void Update();

    SERIALIZE_JSON_STRUCT(AdvancedSettings, buffer_size, filter2d_smooth, contrast, renorm_constant)
};

} // namespace holovibes