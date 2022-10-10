#pragma once

#include "parameters_handler.hh"

#include "cache_icompute.hh"

#include "advanced.hh"
#include "compute.hh"

namespace holovibes
{
using GSHCachesToSync = CachesToSync<CacheICompute>;

template <typename ParamRef>
class CacheGSHSetters;

class CacheGSH : public ParametersHandlerRef<CacheGSH,
                                             CacheGSHSetters<CacheGSH>,
                                             GSHCachesToSync,

                                             BatchSize,
                                             DivideConvolutionEnable,
                                             Lambda,
                                             DisplayRate,
                                             InputBufferSize,
                                             TimeStride>
{
};

template <typename ParamRef>
class CacheGSHSetters
{
  public:
    template <typename T>
    void setter(ParamRef& ref, T& old_value, T&& new_value)
    {
        ref.default_setter(old_value, std::forward<T>(new_value));
    }

    template <>
    void setter<BatchSize>(ParamRef& ref, BatchSize& old_value, BatchSize&& new_value)
    {
        int value = new_value.get_value();
        // FIXME : need all vars on ParametersHandler
        if (value > ref.template get_value<InputBufferSize>())
            value = ref.template get_value<InputBufferSize>();
        if (ref.template get_value<TimeStride>() < value)
            ref.template set_value<TimeStride>(TimeStride{value});
        // Go to lower multiple
        if (ref.template get_value<TimeStride>() % value != 0)
            ref.template set_value<TimeStride>(
                TimeStride{ref.template get_value<TimeStride>() - ref.template get_value<TimeStride>() % value});

        old_value = std::forward<BatchSize>(BatchSize{new_value});
    }

    template <>
    void setter<TimeStride>(ParamRef& ref, TimeStride& old_value, TimeStride&& new_value)
    {
        // FIXME: temporary fix due to ttstride change in pipe.make_request
        // std::lock_guard<std::mutex> lock(mutex_);
        int value = new_value.get_value();

        if (ref.template get_value<BatchSize>() > value)
            ref.template set_value<TimeStride>(ref.template get_value<BatchSize>());
        // Go to lower multiple
        if (value % ref.template get_value<BatchSize>() != 0)
            ref.template set_value<TimeStride>(value - value % ref.template get_value<BatchSize>());

        old_value = std::forward<TimeStride>(TimeStride{new_value});
    }
};
} // namespace holovibes
