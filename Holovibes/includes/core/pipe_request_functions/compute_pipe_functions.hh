#pragma once

#include "pipe.hh"
#include "logger.hh"
#include "micro_cache.hh"

#include "compute_struct.hh"

namespace holovibes
{
class ComputePipeRequest
{
  public:
    template <typename T>
    void operator()(typename T::ConstRefType, typename T::ConstRefType, Pipe& pipe)
    {
    }

    template <>
    void operator()<BatchSize>(int new_value, int old_value, Pipe& pipe)
    {
        LOG_DEBUG(compute_worker, "UPDATE BatchSize");

        pipe.update_spatial_transformation_parameters();
        pipe.get_gpu_input_queue().resize(new_value.get_value());
    }

    template <>
    void operator()<TimeTransformationSize>(uint new_value, uint old_value, Pipe& pipe)
    {
        LOG_DEBUG(compute_worker, "UPDATE TimeTransformationSize");

        if (!pipe.update_time_transformation_size(compute_cache_.get_value<TimeTransformationSize>()))
        {
            // WTF
            success_allocation = false;

            GSH::instance().change_value<ViewAccuP>()->set_index(0);
            GSH::instance().set_value<TimeTransformationSize>(1);
            update_time_transformation_size(1);
            LOG_WARN(compute_worker, "Updating #img failed; #img updated to 1");
        }
    }

    template <>
    void operator()<Convolution>(const ConvolutionStruct& new_value, const ConvolutionStruct& old_value, Pipe& pipe)
    {
        if (new_value.get_is_enabled() == old_value.get_is_enabled())
        {
            LOG_TRACE(compute_worker, "UPDATE Convolution : Nothing to do");
            return;
        }

        if (new_value.get_is_enabled() == false)
        {
            LOG_DEBUG(compute_worker, "UPDATE Convolution : disable ");
            postprocess_->dispose();
        }
        else if (new_value.get_is_enabled() == true)
        {
            LOG_DEBUG(compute_worker, "UPDATE Convolution : enable");
            postprocess_->init();
        }
    }
};
} // namespace holovibes
