/*! \file advanced_struct.hh
 *
 * \brief Advanced Struct
 *
 */

#pragma once

#include "all_struct.hh"
#include <optional>

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

        /*! \brief Will be expanded into `to_json` and `from_json` functions. */
        SERIALIZE_JSON_STRUCT(BufferSizes, input, file, record, output, time_transformation_cuts);

        /*!
         * \brief Will be expanded into `Load`, `Update` and `Assert` functions that respectivly
         * synchronize variables of BufferSizes with the one in GSH, update variables of GSH
         * with the one of BufferSizes and assert that the BufferSizes variables are valid
         */
        SETTING_RELATED_FUNCTIONS();
    };

    /*! \class Filter2DSmooth
     *
     * \brief Class that represents Filter2DSmooth
     */
    struct Filter2DSmooth
    {
        int low = 0;
        int high = 0;

        /*! \brief Will be expanded into `to_json` and `from_json` functions. */
        SERIALIZE_JSON_STRUCT(Filter2DSmooth, low, high);

        /*!
         * \brief Will be expanded into `Load`, `Update` and `Assert` functions that respectivly
         * synchronize variables of Filter2DSmooth with the one in GSH, update variables of GSH
         * with the one of Filter2DSmooth and assert that the Filter2DSmooth variables are valid
         */
        SETTING_RELATED_FUNCTIONS();
    };

    /*! \class ContrastThreshold
     *
     * \brief Class that represents ContrastThreshold
     */
    struct ContrastThreshold
    {
        float lower = 0.02f;
        float upper = 99.8f;
        unsigned frame_index_offset = 2;

        /*! \brief Will be expanded into `to_json` and `from_json` functions. */
        SERIALIZE_JSON_STRUCT(ContrastThreshold, lower, upper, frame_index_offset);

        /*!
         * \brief Will be expanded into `Load`, `Update` and `Assert` functions that respectivly
         * synchronize variables of ContrastThreshold with the one in GSH, update variables of GSH
         * with the one of ContrastThreshold and assert that the ContrastThreshold variables are valid
         */
        SETTING_RELATED_FUNCTIONS();
    };

    BufferSizes buffer_size;
    Filter2DSmooth filter2d_smooth;
    ContrastThreshold contrast;
    unsigned renorm_constant = 5;
    unsigned int raw_bitshift = 0;
    unsigned int nb_frames_to_record = 0;

    /*! \brief Will be expanded into `to_json` and `from_json` functions. */
    SERIALIZE_JSON_STRUCT(
        AdvancedSettings, buffer_size, filter2d_smooth, contrast, renorm_constant, raw_bitshift, nb_frames_to_record);

    /*!
     * \brief Will be expanded into `Load`, `Update` and `Assert` functions that respectivly
     * synchronize variables of AdvancedSettings with the one in GSH, update variables of GSH
     * with the one of AdvancedSettings and assert that the AdvancedSettings variables are valid
     */
    SETTING_RELATED_FUNCTIONS();
};

} // namespace holovibes
