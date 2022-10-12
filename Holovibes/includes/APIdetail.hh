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

#include "micro_cache_tmp.hh"

#include <nlohmann/json.hpp>
using json = ::nlohmann::json;

namespace holovibes::api
{

template <typename T>
typename T::ValueType get_value()
{
    return GSH::instance().get_value<T>();
}

// Setters
template <typename T>
void set_value(typename T::ValueConstRef value)
{
    GSH::instance().set_value<T>(value);
}

template <>
void set_value<BatchSize>(int value);
template <>
void set_value<TimeStride>(int value);

template <>
inline void set_value<BatchSize>(int value)
{
    GSH::instance().set_value<BatchSize>(value);

    // FIXME : need all vars on MicroCacheTmp
    if (value > get_value<InputBufferSize>())
        GSH::instance().set_value<BatchSize>(value);

    if (get_value<TimeStride>() < value)
        set_value<TimeStride>(value);
    // Go to lower multiple
    if (get_value<TimeStride>() % value != 0)
        set_value<TimeStride>(get_value<TimeStride>() - get_value<TimeStride>() % value);
}

template <>
inline void set_value<TimeStride>(int value)
{
    // FIXME: temporary fix due to ttstride change in pipe.make_request
    // std::lock_guard<std::mutex> lock(mutex_);
    GSH::instance().set_value<TimeStride>(value);

    if (get_value<BatchSize>() > value)
        return set_value<TimeStride>(get_value<BatchSize>());

    // Go to lower multiple
    if (value % get_value<BatchSize>() != 0)
        return set_value<TimeStride>(value - value % get_value<BatchSize>());
}
} // namespace holovibes::api
