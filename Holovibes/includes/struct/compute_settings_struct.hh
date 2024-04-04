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
#include "API.hh"

namespace holovibes
{
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

    /*! \brief Synchornize instance of ComputeSettings with GSH */
    void Update();
    /*! \brief Synchornize instance of GSH with ComputeSettings */
    void Load();
    /*! \brief Debug function - Dump instance of ComputeSettings in json */
    void Dump(const std::string& filename);
    /*! \brief Assert that the ComputeSettings are valid, if in cli: raise an error when non valid, else: use the default ones from the app. */
    void Assert(bool cli) const;

    SERIALIZE_JSON_STRUCT(ComputeSettings, image_rendering, view, color_composite_image, advanced)
};
} // namespace holovibes
