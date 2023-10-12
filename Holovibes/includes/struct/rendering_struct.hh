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

        void Update();
        void Load();

        SERIALIZE_JSON_STRUCT(Filter2D, enabled, inner_radius, outer_radius)
    };

    /*! \class Filter
     *
     * \brief Class that represents Input Filter
     */
    struct Filter
    {
        bool enabled = false;
        std::string type;

        void Update();
        void Load();

        SERIALIZE_JSON_STRUCT(Filter, enabled, type);
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

        void Update();
        void Load();

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
    float propagation_distance = 1.5f;
    Convolution convolution;
    Filter input_filter;

    void Update();
    void Load();

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
                          convolution)
};

} // namespace holovibes
