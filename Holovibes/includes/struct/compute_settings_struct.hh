/*! \file
 *
 * \brief ComputeSettings Struct
 *
 */

#pragma once

#include "rendering_struct.hh"
#include "view_struct.hh"
#include "composite_struct.hh"
#include "advanced_struct.hh"

namespace holovibes
{

/*! \enum ComputeSettingsVersion
 *
 * \brief List of all possible version of compute settings
 */
enum class ComputeSettingsVersion
{
    V2,
    V3,
    V4,
    V5
};

/*! \class ComputeSettings
 *
 * \brief Class that represents the footer of holo file
 */
struct ComputeSettings
{
    Rendering image_rendering;
    Views view;
    Composite color_composite_image;
    AdvancedSettings advanced;

    /*! \brief Debug function - Dump instance of ComputeSettings in json */
    void Dump(const std::string& filename);

    /*! \brief Converts a json from the version `from` to the current version */
    static void convert_json(json& data, ComputeSettingsVersion from);

    /*! \brief Will be expanded into `to_json` and `from_json` functions. */
    SERIALIZE_JSON_STRUCT(ComputeSettings, image_rendering, view, color_composite_image, advanced);

    /*!
     * \brief Will be expanded into `Load`, `Update` and `Assert` functions that respectivly
     * synchronize variables of ComputeSettings with the one in GSH, update variables of GSH
     * with the one of ComputeSettings and assert that the ComputeSettings variables are valid
     */
    SETTING_RELATED_FUNCTIONS();
};
} // namespace holovibes
