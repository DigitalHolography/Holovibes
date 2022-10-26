#pragma once

#include "API.hh"

namespace holovibes
{
template <>
void ComputePipeRequestOnSync::operator()<BatchSize>(int new_value, int old_value, Pipe& pipe)
{
    LOG_TRACE(compute_worker, "UPDATE BatchSize");

    pipe.update_spatial_transformation_parameters();
    pipe.get_gpu_input_queue().resize(new_value.get_value());
}

template <>
void ComputePipeRequestOnSync::operator()<TimeStride>(int new_value, int old_value, Pipe& pipe)
{
    LOG_TRACE(compute_worker, "UPDATE TimeStride");

    batch_env_.batch_index = 0;
}

template <>
void ComputePipeRequestOnSync::operator()<TimeTransformationSize>(uint new_value, uint old_value, Pipe& pipe)
{
    LOG_TRACE(compute_worker, "UPDATE TimeTransformationSize");

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
void ComputePipeRequestOnSync::operator()<Convolution>(const ConvolutionStruct& new_value,
                                                       const ConvolutionStruct& old_value,
                                                       Pipe& pipe)
{
    LOG_TRACE(compute_worker, "UPDATE Convolution");

    if (new_value.get_is_enabled() == old_value.get_is_enabled())
        return;

    if (new_value.get_is_enabled() == false)
        postprocess_->dispose();
    else if (new_value.get_is_enabled() == true)
        postprocess_->init();
}

template <>
void ComputePipeRequestOnSync::operator()<TimeTransformationCuts>(bool new_value, bool old_value, Pipe& pipe)
{
    LOG_TRACE(compute_worker, "UPDATE Convolution");

    if (new_value == false)
        pipe.dispose_cuts();
    else if (new_value == true)
        pipe.init_cuts();
}
} // namespace holovibes
