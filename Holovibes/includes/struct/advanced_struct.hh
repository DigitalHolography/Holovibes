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

    bool operator!=(const Filter2DSmoothStruct& rhs) { return low != rhs.low || high != rhs.high; }
};

/*! \class ContrastThreshold
 *
 * \brief Class that represents ContrastThreshold
 */
struct ContrastThresholdStruct
{
    float lower = 0.5f;
    float upper = 99.5f;
    unsigned frame_index_offset = 2;

    SERIALIZE_JSON_STRUCT(ContrastThresholdStruct, lower, upper, frame_index_offset)

    bool operator!=(const ContrastThresholdStruct& rhs)
    {
        return lower != rhs.lower || upper != rhs.upper || frame_index_offset != rhs.frame_index_offset;
    }
};

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

/*! \class AdvancedSettings
 *
 * \brief Class that represents the advanced cache
 */
struct AdvancedSettings
{
    BufferSize buffer_size;
    Filter2DSmoothStruct filter2d_smooth;
    ContrastThresholdStruct contrast;
    unsigned renorm_constant = 5;
    unsigned int raw_bitshift = 0;

    void Update();
    void Load();

    SERIALIZE_JSON_STRUCT(AdvancedSettings, buffer_size, filter2d_smooth, contrast, renorm_constant, raw_bitshift)
};

inline std::ostream& operator<<(std::ostream& os, const Filter2DSmoothStruct& value) { return os << json{value}; }
inline std::ostream& operator<<(std::ostream& os, const ContrastThresholdStruct& value) { return os << json{value}; }
inline std::ostream& operator<<(std::ostream& os, const BufferSize& value) { return os << json{value}; }
inline std::ostream& operator<<(std::ostream& os, const AdvancedSettings& value) { return os << json{value}; }
} // namespace holovibes
