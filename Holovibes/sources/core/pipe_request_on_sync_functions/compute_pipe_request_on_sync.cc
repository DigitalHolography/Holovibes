#include "API.hh"

namespace holovibes
{
template <>
void ComputePipeRequestOnSync::operator()<BatchSize>(int new_value, Pipe& pipe)
{
    LOG_UPDATE_PIPE(BatchSize);

    pipe.update_spatial_transformation_parameters();
    pipe.get_gpu_input_queue().resize(new_value);

    pipe.get_export_cache().virtual_synchronize_W<FrameRecordMode, ExportPipeRequestOnSync>(pipe);
    pipe.get_export_cache().virtual_synchronize_W<ChartRecord, ExportPipeRequestOnSync>(pipe);
    pipe.get_composite_cache().virtual_synchronize_W<CompositeRGB, CompositePipeRequestOnSync>(pipe);
    pipe.get_composite_cache().virtual_synchronize_W<CompositeHSV, CompositePipeRequestOnSync>(pipe);
}

template <>
void ComputePipeRequestOnSync::operator()<TimeStride>(int new_value, Pipe& pipe)
{
    LOG_UPDATE_PIPE(TimeStride);

    pipe.get_batch_env().batch_index = 0;
}

template <>
void ComputePipeRequestOnSync::operator()<TimeTransformationSize>(uint new_value, Pipe& pipe)
{
    LOG_UPDATE_PIPE(TimeTransformationSize);

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
void ComputePipeRequestOnSync::on_sync<Convolution>(const ConvolutionStruct& new_value,
                                                    const ConvolutionStruct& old_value,
                                                    Pipe& pipe)
{
    if (new_value.enabled != old_value.enabled)
    {
        operator()<Convolution>(new_value, pipe);
    }
}

template <>
void ComputePipeRequestOnSync::operator()<Convolution>(const ConvolutionStruct& new_value, Pipe& pipe)
{
    LOG_UPDATE_PIPE(Convolution);

    if (new_value.enabled == false)
        pipe.get_postprocess().dispose();
    else if (new_value.enabled == true)
        pipe.get_postprocess().init();

    request_pipe_refresh();
}

template <>
void ComputePipeRequestOnSync::operator()<TimeTransformationCutsEnable>(bool new_value, Pipe& pipe)
{
    LOG_UPDATE_PIPE(Convolution);

    if (new_value == false)
        pipe.dispose_cuts();
    else if (new_value == true)
        pipe.init_cuts();
}
} // namespace holovibes
