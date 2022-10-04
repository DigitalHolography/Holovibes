#pragma once

#include "pipe.hh"
#include "logger.hh"
#include "parameters_handler.hh"
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
    void call(const T&, Pipe& pipe)
    {
    }

    template <>
    void call<BatchSize>(const BatchSize& batch_size, Pipe& pipe)
    {
        LOG_DEBUG(compute_worker, "UPDATE BATCH");

        pipe.update_spatial_transformation_parameters();
        pipe.get_gpu_input_queue().resize(batch_size.get_value());
    }
};
} // namespace holovibes
