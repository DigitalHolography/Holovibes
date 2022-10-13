#pragma once

#include <optional>

#include "logger.hh"
#include "input_frame_file.hh"
#include "input_frame_file_factory.hh"
#include "holovibes.hh"
#include "view_panel.hh"
#include "AdvancedSettingsWindow.hh"
#include "holovibes_config.hh"
#include "user_interface_descriptor.hh"
#include "global_state_holder.hh"

#include <nlohmann/json.hpp>
using json = ::nlohmann::json;

namespace holovibes::api::detail
{

template <typename T>
typename T::ValueConstRef get_value()
{
    return GSH::instance().get_value<T>();
}

// Setters
template <typename T>
void set_value(typename T::ValueConstRef value)
{
    GSH::instance().set_value<T>(value);
}

} // namespace holovibes::api::detail