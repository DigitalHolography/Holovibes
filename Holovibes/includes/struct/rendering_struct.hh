/*! \file
 *
 * \brief Rendering  Struct
 *
 */

#pragma once

#include "enum_space_transformation.hh"
#include "enum_time_transformation.hh"
#include "enum_computation.hh"
#include "all_struct.hh"

namespace holovibes
{

/*! \class Rendering
 *
 * \brief Class that represents the rendering cache
 */
struct Rendering
{

    /*! \class Filter2D
     *
     * \brief Class that represents Filter2D
     */
    struct Filter2D
    {
        bool enabled = false;
        int inner_radius = 0;
        int outer_radius = 1;

        /*! \brief Will be expanded into `to_json` and `from_json` functions. */
        SERIALIZE_JSON_STRUCT(Filter2D, enabled, inner_radius, outer_radius);

        /*!
         * \brief Will be expanded into `Load`, `Update` and `Assert` functions that respectivly
         * synchronize variables of Filter2D with the one in GSH, update variables of GSH
         * with the one of Filter2D and assert that the Filter2D variables are valid
         */
        SETTING_RELATED_FUNCTIONS();
    };

    /*! \class Filter
     *
     * \brief Class that represents Input Filter
     */
    struct Filter
    {
        std::string type;

        /*! \brief Will be expanded into `to_json` and `from_json` functions. */
        SERIALIZE_JSON_STRUCT(Filter, type);

        /*!
         * \brief Will be expanded into `Load`, `Update` and `Assert` functions that respectivly
         * synchronize variables of Filter with the one in GSH, update variables of GSH
         * with the one of Filter and assert that the Filter variables are valid
         */
        SETTING_RELATED_FUNCTIONS();
    };

    /*! \class Convolution
     *
     * \brief Class that represents Convolution
     */
    struct Convolution
    {
        bool enabled = false;
        std::string type;
        bool divide = false;

        /*! \brief Will be expanded into `to_json` and `from_json` functions. */
        SERIALIZE_JSON_STRUCT(Convolution, enabled, type, divide);

        /*!
         * \brief Will be expanded into `Load`, `Update` and `Assert` functions that respectivly
         * synchronize variables of Convolution with the one in GSH, update variables of GSH
         * with the one of Convolution and assert that the Convolution variables are valid
         */
        SETTING_RELATED_FUNCTIONS();
    };

    Computation image_mode = Computation::Raw;
    unsigned batch_size = 1;
    unsigned time_transformation_stride = 1;
    Filter2D filter2d;
    SpaceTransformation space_transformation = SpaceTransformation::NONE;
    TimeTransformation time_transformation = TimeTransformation::NONE;
    unsigned time_transformation_size = 1;
    float lambda = 852e-9f;
    float propagation_distance = 1.5f;
    Convolution convolution;
    Filter input_filter;

    /*! \brief Will be expanded into `to_json` and `from_json` functions. */
    SERIALIZE_JSON_STRUCT(Rendering,
                          input_filter,
                          image_mode,
                          batch_size,
                          time_transformation_stride,
                          filter2d,
                          space_transformation,
                          time_transformation,
                          time_transformation_size,
                          lambda,
                          propagation_distance,
                          convolution);

    /*!
     * \brief Will be expanded into `Load`, `Update` and `Assert` functions that respectivly
     * synchronize variables of Rendering with the one in GSH, update variables of GSH
     * with the one of Rendering and assert that the Rendering variables are valid
     */
    SETTING_RELATED_FUNCTIONS();
};

} // namespace holovibes
