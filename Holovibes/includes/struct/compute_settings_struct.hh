#pragma once

#include "rendering_struct.hh"
#include "view_struct.hh"
#include "composite_struct.hh"
#include "advanced_struct.hh"
#include "API.hh"

namespace holovibes
{
struct ComputeSettings
{
    Rendering image_rendering;
    Views view;
    Composite composite;
    AdvancedSettings advanced;

    void Update();
    void Load();
    void Dump(const std::string& filename);

    SERIALIZE_JSON_STRUCT(ComputeSettings, image_rendering, view, composite, advanced)
};
} // namespace holovibes