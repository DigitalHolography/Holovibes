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
        std::string type;
        bool divide = false;

        /*! \brief Will be expanded into `to_json` and `from_json` functions. */
        SERIALIZE_JSON_STRUCT(Convolution, type, divide);

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
    unsigned lsci_spatial_window = 5;
    // Local PCA defaults: 32x32 tiles, 16x16 spatial strides (four contributions
    // per pixel when frame dimensions are divisible by 16).
    unsigned pca_width_x = 32;
    unsigned pca_width_y = 32;
    unsigned pca_stride_x = 16;
    unsigned pca_stride_y = 16;
    float pca_rank_factor = 1.f;
    float lambda = 852e-9f;
    float propagation_distance = 1.5f;
    Convolution convolution;
    Filter input_filter;

    /*! \brief Serialize rendering settings while keeping the optional LSCI window backward compatible. */
    friend void to_json(json& j, const Rendering& value)
    {
        j = json{{"input_filter", value.input_filter},
                 {"image_mode", value.image_mode},
                 {"batch_size", value.batch_size},
                 {"time_transformation_stride", value.time_transformation_stride},
                 {"filter2d", value.filter2d},
                 {"space_transformation", value.space_transformation},
                 {"time_transformation", value.time_transformation},
                 {"time_transformation_size", value.time_transformation_size},
                 {"lambda", value.lambda},
                 {"propagation_distance", value.propagation_distance},
                 {"convolution", value.convolution}};
        if (value.time_transformation == TimeTransformation::LSCI || value.lsci_spatial_window != 5)
            j["lsci_spatial_window"] = value.lsci_spatial_window;
        // Optional keys keep existing algorithm configurations backward compatible.
        if (value.time_transformation == TimeTransformation::LOCAL_PCA || value.pca_width_x != 32 ||
            value.pca_width_y != 32 || value.pca_stride_x != 16 || value.pca_stride_y != 16 ||
            value.pca_rank_factor != 1.f)
        {
            j["pca_width_x"] = value.pca_width_x;
            j["pca_width_y"] = value.pca_width_y;
            j["pca_stride_x"] = value.pca_stride_x;
            j["pca_stride_y"] = value.pca_stride_y;
            j["pca_rank_factor"] = value.pca_rank_factor;
        }
    }

    friend void from_json(const json& j, Rendering& value)
    {
        j.at("input_filter").get_to(value.input_filter);
        j.at("image_mode").get_to(value.image_mode);
        j.at("batch_size").get_to(value.batch_size);
        j.at("time_transformation_stride").get_to(value.time_transformation_stride);
        j.at("filter2d").get_to(value.filter2d);
        j.at("space_transformation").get_to(value.space_transformation);
        j.at("time_transformation").get_to(value.time_transformation);
        j.at("time_transformation_size").get_to(value.time_transformation_size);
        // Recordings and configuration files written before LSCI use the paper's 5x5 neighborhood.
        value.lsci_spatial_window = j.value("lsci_spatial_window", 5u);
        value.pca_width_x = j.value("pca_width_x", 32u);
        value.pca_width_y = j.value("pca_width_y", 32u);
        value.pca_stride_x = j.value("pca_stride_x", 16u);
        value.pca_stride_y = j.value("pca_stride_y", 16u);
        // Settings written before the rank slider preserve the original cutoff.
        value.pca_rank_factor = j.value("pca_rank_factor", 1.f);
        j.at("lambda").get_to(value.lambda);
        j.at("propagation_distance").get_to(value.propagation_distance);
        j.at("convolution").get_to(value.convolution);
    }

    /*!
     * \brief Will be expanded into `Load`, `Update` and `Assert` functions that respectivly
     * synchronize variables of Rendering with the one in GSH, update variables of GSH
     * with the one of Rendering and assert that the Rendering variables are valid
     */
    SETTING_RELATED_FUNCTIONS();
};

} // namespace holovibes
