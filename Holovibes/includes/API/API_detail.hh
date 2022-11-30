#pragma once

#include "logger.hh"

#include "holovibes_config.hh"

#include "input_frame_file.hh"
#include "input_frame_file_factory.hh"

#include "all_caches.hh"
#include "holovibes.hh"

#include "global_state_holder.hh"

#include <nlohmann/json.hpp>

using json = ::nlohmann::json;

namespace holovibes::api::detail
{

template <typename T>
typename T::ConstRefType get_value()
{
    return GSH::instance().get_value<T>();
}

template <typename T>
void set_value(typename T::ConstRefType value)
{
    GSH::instance().set_value<T>(value);
}

template <typename T>
TriggerChangeValue<typename T::ValueType> change_value()
{
    return GSH::instance().change_value<T>();
}

} // namespace holovibes::api::detail
