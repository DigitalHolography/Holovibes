/*! \file
 *
 * \brief Advanced Struct
 *
 */

#pragma once

#include "all_struct.hh"

namespace holovibes
{

/*! \class AdvancedSettings
 *
 * \brief Class that represents the advanced cache
 */
struct AdvancedSettings
{
    /*! \class BufferSizes
     *
     * \brief Class that represents BufferSizes
     */
    struct BufferSizes
    {
        unsigned input = 512;
        unsigned file = 512;
        unsigned record = 1024;
        unsigned output = 256;
        unsigned time_transformation_cuts = 512;

        void Load();
        void Update();

        SERIALIZE_JSON_STRUCT(BufferSizes, input, file, record, output, time_transformation_cuts)
    };

    /*! \class Filter2DSmooth
     *
     * \brief Class that represents Filter2DSmooth
     */
    struct Filter2DSmooth
    {
        int low = 0;
        int high = 0;

        void Load();
        void Update();

        SERIALIZE_JSON_STRUCT(Filter2DSmooth, low, high)
    };

    /*! \class ContrastThreshold
     *
     * \brief Class that represents ContrastThreshold
     */
    struct ContrastThreshold
    {
        float lower = 0.5f;
        float upper = 99.5f;
        unsigned frame_index_offset = 2;

        void Load();
        void Update();

        SERIALIZE_JSON_STRUCT(ContrastThreshold, lower, upper, frame_index_offset)
    };

    BufferSizes buffer_size;
    Filter2DSmooth filter2d_smooth;
    ContrastThreshold contrast;
    unsigned renorm_constant = 5;
    unsigned int raw_bitshift = 0;

    void Update();
    void Load();

    SERIALIZE_JSON_STRUCT(AdvancedSettings, buffer_size, filter2d_smooth, contrast, renorm_constant, raw_bitshift)
};

} // namespace holovibes
