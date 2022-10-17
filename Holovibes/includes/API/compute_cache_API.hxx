#pragma once

#include "compute_cache_API.hh"
#include "advanced_cache_API.hh"

namespace holovibes::api
{

inline void set_batch_size(int value)
{
    GSH::instance().set_value<BatchSize>(value);

    // FIXME : need all vars on MicroCache
    if (value > get_input_buffer_size())
        GSH::instance().set_value<BatchSize>(value);

    if (get_time_stride() < value)
        set_time_stride(value);
    // Go to lower multiple
    if (get_time_stride() % value != 0)
        set_time_stride(get_time_stride() - get_time_stride() % value);
}

inline void set_time_stride(int value)
{
    // FIXME: temporary fix due to ttstride change in pipe.make_request
    // std::lock_guard<std::mutex> lock(mutex_);
    GSH::instance().set_value<TimeStride>(value);

    if (get_batch_size() > value)
        return set_time_stride(get_batch_size());

    // Go to lower multiple
    if (value % get_batch_size() != 0)
        return set_time_stride(value - value % get_batch_size());
}

} // namespace holovibes::api
