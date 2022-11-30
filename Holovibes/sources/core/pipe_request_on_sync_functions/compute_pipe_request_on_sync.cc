#include "API.hh"

namespace holovibes
{
template <>
void ComputePipeRequestOnSync::operator()<BatchSize>(int new_value, Pipe& pipe)
{
    LOG_UPDATE_ON_SYNC(BatchSize);

    pipe.update_spatial_transformation_parameters();
    // NOP
    pipe.get_gpu_input_queue().resize(new_value);

    request_pipe_refresh();
}

template <>
void ComputePipeRequestOnSync::operator()<TimeStride>(int new_value, Pipe& pipe)
{
    LOG_UPDATE_ON_SYNC(TimeStride);

    pipe.get_batch_env().batch_index = 0;
}

template <>
void ComputePipeRequestOnSync::operator()<TimeTransformationSize>(uint new_value, Pipe& pipe)
{
    LOG_UPDATE_ON_SYNC(TimeTransformationSize);

    if (!pipe.update_time_transformation_size(pipe.get_compute_cache().get_value<TimeTransformationSize>()))
    {
        request_fail();

        api::detail::change_value<ViewAccuP>()->start = 0;
        api::detail::set_value<TimeTransformationSize>(1);
        pipe.update_time_transformation_size(1);
        LOG_WARN("Updating #img failed; #img updated to 1");
    }
}

template <>
void ComputePipeRequestOnSync::operator()<Convolution>(const ConvolutionStruct& new_value, Pipe& pipe)
{
    LOG_UPDATE_ON_SYNC(Convolution);

    if (new_value.enabled == false)
        pipe.get_postprocess().dispose();
    else if (new_value.enabled == true)
        pipe.get_postprocess().init();

    pipe.get_view_cache().virtual_synchronize_W<ViewXY, ViewPipeRequestOnSync>(pipe);
    pipe.get_view_cache().virtual_synchronize_W<ViewXZ, ViewPipeRequestOnSync>(pipe);
    pipe.get_view_cache().virtual_synchronize_W<ViewYZ, ViewPipeRequestOnSync>(pipe);
    request_pipe_refresh();
}

template <>
void ComputePipeRequestOnSync::operator()<TimeTransformationCutsEnable>(bool new_value, Pipe& pipe)
{
    LOG_UPDATE_ON_SYNC(TimeTransformationCutsEnable);

    if (new_value == false)
        pipe.dispose_cuts();
    else if (new_value == true)
        pipe.init_cuts();

    pipe.get_view_cache().virtual_synchronize_W<ViewXZ, ViewPipeRequestOnSync>(pipe);
    pipe.get_view_cache().virtual_synchronize_W<ViewYZ, ViewPipeRequestOnSync>(pipe);

    request_pipe_refresh();
}

} // namespace holovibes
