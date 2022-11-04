/*! \file
 *
 * \brief Advanced Struct
 *
 */

#pragma once

#include "all_struct.hh"

namespace holovibes
{

/*! \class Filter2DSmooth
 *
 * \brief Class that represents Filter2DSmooth
 */
struct Filter2DSmoothStruct
{
    int low = 0;
    int high = 0;

    SERIALIZE_JSON_STRUCT(Filter2DSmoothStruct, low, high)
};

/*! \class ContrastThreshold
 *
 * \brief Class that represents ContrastThreshold
 */
struct ContrastThresholdStruct
{
    float lower = 0.5f;
    float upper = 99.5f;
    unsigned cuts_p_offset = 2;

    SERIALIZE_JSON_STRUCT(ContrastThresholdStruct, lower, upper, cuts_p_offset)
};

/*! \class AdvancedSettings
 *
 * \brief Class that represents the advanced cache
 */
struct AdvancedSettings
{
    /*! \class BufferSize
     *
     * \brief Class that represents BufferSize
     */
    struct BufferSize
    {
        unsigned input = 512;
        unsigned file = 512;
        unsigned record = 1024;
        unsigned output = 256;
        unsigned time_transformation_cuts = 512;

        void Load();
        void Update();

        SERIALIZE_JSON_STRUCT(BufferSize, input, file, record, output, time_transformation_cuts)
    };

    BufferSize buffer_size;
    Filter2DSmoothStruct filter2d_smooth;
    ContrastThresholdStruct contrast;
    unsigned renorm_constant = 5;
    unsigned int raw_bitshift = 0;

    void Update();
    void Load();

    SERIALIZE_JSON_STRUCT(AdvancedSettings, buffer_size, filter2d_smooth, contrast, renorm_constant, raw_bitshift)
};

} // namespace holovibes