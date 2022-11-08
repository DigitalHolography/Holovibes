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
#include "all_struct.hh"

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
    Composite composite;
    AdvancedSettings advanced;

    /*! \brief Synchornize instance of ComputeSettings with GSH */
    void Update();
    /*! \brief Synchornize instance of GSH with ComputeSettings */
    void Load();
    /*! \brief Debug function - Dump instance of ComputeSettings in json */
    void Dump(const std::string& filename);

    SERIALIZE_JSON_STRUCT(ComputeSettings, image_rendering, view, composite, advanced)
};

inline std::ostream& operator<<(std::ostream& os, const ComputeSettings& value) { return os << json{value}; }

} // namespace holovibes