#include "API.hh"

namespace holovibes
{
template <>
void ComputePipeRequestOnSync::operator()<BatchSize>(int new_value, int old_value, Pipe& pipe)
{
    LOG_TRACE(compute_worker, "UPDATE BatchSize");

    pipe.update_spatial_transformation_parameters();
    pipe.get_gpu_input_queue().resize(new_value);
}

template <>
void ComputePipeRequestOnSync::operator()<TimeStride>(int new_value, int old_value, Pipe& pipe)
{
    LOG_TRACE(compute_worker, "UPDATE TimeStride");

    pipe.get_batch_env().batch_index = 0;
}

template <>
void ComputePipeRequestOnSync::operator()<TimeTransformationSize>(uint new_value, uint old_value, Pipe& pipe)
{
    LOG_TRACE(compute_worker, "UPDATE TimeTransformationSize");

    if (!pipe.update_time_transformation_size(pipe.get_compute_cache().get_value<TimeTransformationSize>()))
    {
        request_fail();

        GSH::instance().change_value<ViewAccuP>()->index = 0;
        GSH::instance().set_value<TimeTransformationSize>(1);
        pipe.update_time_transformation_size(1);
        LOG_WARN(compute_worker, "Updating #img failed; #img updated to 1");
    }
}

template <>
void ComputePipeRequestOnSync::operator()<Convolution>(const ConvolutionStruct& new_value,
                                                       const ConvolutionStruct& old_value,
                                                       Pipe& pipe)
{
    LOG_TRACE(compute_worker, "UPDATE Convolution");

    if (new_value.enabled != old_value.enabled)
    {
        if (new_value.enabled == false)
            pipe.get_postprocess().dispose();
        else if (new_value.enabled == true)
            pipe.get_postprocess().init();
    }

    request_pipe_refresh();
}

template <>
void ComputePipeRequestOnSync::operator()<TimeTransformationCutsEnable>(bool new_value, bool old_value, Pipe& pipe)
{
    LOG_TRACE(compute_worker, "UPDATE Convolution");

    if (new_value == false)
        pipe.dispose_cuts();
    else if (new_value == true)
        pipe.init_cuts();
}
} // namespace holovibes
