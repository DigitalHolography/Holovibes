#pragma once

#include "detail.hh"

namespace holovibes
{
class ComputePipeRequest : public PipeRequestFunctions
{
  public:
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
