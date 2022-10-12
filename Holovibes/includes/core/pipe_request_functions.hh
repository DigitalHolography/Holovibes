#pragma once

#include "pipe.hh"
#include "logger.hh"
#include "micro_cache_tmp.hh"
#include "on_synchronize_functions.hh"

namespace holovibes
{
class PipeRequestFunctions
{
  public:
    using BeforeMethods = OnSynchronizeFunctions;

  public:
    template <typename T>
    bool test(const T& value)
    {
        return value.get_has_been_synchronized();
    }

    template <typename T>
    void operator()(const T&, Pipe& pipe)
    {
    }

    template <>
    void operator()<BatchSize>(const BatchSize& batch_size, Pipe& pipe)
    {
        LOG_DEBUG(compute_worker, "UPDATE BATCH");

        pipe.update_spatial_transformation_parameters();
        pipe.get_gpu_input_queue().resize(batch_size.get_value());
    }
};
} // namespace holovibes
