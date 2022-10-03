#pragma once

#include "rendering_struct.hh"
#include "view_struct.hh"
#include "composite_struct.hh"
#include "advanced_struct.hh"

namespace holovibes
{
struct ComputeSettings
{
    Rendering image_rendering;
    Views view;
    Composite composite;
    AdvancedSettings advanced;

    SERIALIZE_JSON_STRUCT(ComputeSettings, image_rendering, view, composite, advanced)
};
} // namespace holovibes