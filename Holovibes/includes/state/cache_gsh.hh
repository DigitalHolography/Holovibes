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
                                             DisplayRate>
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
        // FIXME : need all vars on ParametersHandler
        // if (value > advanced_cache_.get_input_buffer_size())
        //     value = advanced_cache_.get_input_buffer_size();
        // if (compute_cache_.get_time_stride() < value)
        //     compute_cache_.set_time_stride(value);
        // // Go to lower multiple
        // if (compute_cache_.get_time_stride() % value != 0)
        //     compute_cache_.set_time_stride(compute_cache_.get_time_stride() - compute_cache_.get_time_stride() %
        //     value);

        old_value = std::forward<BatchSize>(new_value);
    }
};
} // namespace holovibes
