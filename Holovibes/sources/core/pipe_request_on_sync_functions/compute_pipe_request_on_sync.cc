#include "API.hh"

namespace holovibes
{
template <>
void ComputePipeRequestOnSync::operator()<BatchSize>(int new_value, Pipe& pipe)
{
    LOG_UPDATE_ON_SYNC(BatchSize);

    pipe.get_gpu_input_queue().set_new_batch_size(new_value);

    const auto& gpu_input_queue_fd = pipe.get_gpu_input_queue().get_fd();
    pipe.get_batch_env().batch_index = 0;
    // We avoid the depth in the multiplication because the resize already take
    // it into account
    pipe.get_buffers().gpu_spatial_transformation_buffer.resize(new_value * gpu_input_queue_fd.get_frame_res());

    long long int n[] = {gpu_input_queue_fd.height, gpu_input_queue_fd.width};

    // This plan has a useful significant memory cost, check XtplanMany comment
    pipe.get_spatial_transformation_plan().XtplanMany(
        2,                                  // 2D
        n,                                  // Dimension of inner most & outer most dimension
        n,                                  // Storage dimension size
        1,                                  // Between two inputs (pixels) of same image distance is one
        gpu_input_queue_fd.get_frame_res(), // Distance between 2 same index pixels of 2 images
        CUDA_C_32F,                         // Input type
        n,
        1,
        gpu_input_queue_fd.get_frame_res(), // Ouput layout same as input
        CUDA_C_32F,                         // Output type
        new_value,                          // Batch size
        CUDA_C_32F);                        // ComputeModeEnum type

    request_pipe_refresh();
}

template <>
void ComputePipeRequestOnSync::operator()<TimeStride>(int new_value, Pipe& pipe)
{
    LOG_UPDATE_ON_SYNC(TimeStride);

    pipe.get_batch_env().batch_index = 0;

    request_pipe_refresh();
}

template <>
void ComputePipeRequestOnSync::operator()<TimeTransformationSize>(uint new_value, Pipe& pipe)
{
    LOG_UPDATE_ON_SYNC(TimeTransformationSize);

    pipe.get_time_transformation_env().gpu_p_acc_buffer.resize(pipe.get_gpu_input_queue().get_fd().get_frame_res() *
                                                               new_value);

    if (pipe.get_compute_cache().get_value<TimeTransformation>() == TimeTransformationEnum::NONE)
        return;

    if (pipe.get_compute_cache().get_value<TimeTransformation>() == TimeTransformationEnum::STFT ||
        pipe.get_compute_cache().get_value<TimeTransformation>() == TimeTransformationEnum::SSA_STFT)
    {
        /* CUFFT plan1d realloc */
        int inembed_stft[1] = {static_cast<int>(new_value)};

        int zone_size = static_cast<int>(pipe.get_gpu_input_queue().get_fd().get_frame_res());

        pipe.get_time_transformation_env()
            .stft_plan
            .planMany(1, inembed_stft, inembed_stft, zone_size, 1, inembed_stft, zone_size, 1, CUFFT_C2C, zone_size);
    }

    if (pipe.get_compute_cache().get_value<TimeTransformation>() == TimeTransformationEnum::PCA ||
        pipe.get_compute_cache().get_value<TimeTransformation>() == TimeTransformationEnum::SSA_STFT)
    {
        // Pre allocate all the buffer only when n changes to avoid 1 allocation
        pipe.get_time_transformation_env().pca_cov.resize(new_value * new_value);
        pipe.get_time_transformation_env().pca_eigen_values.resize(new_value);
        pipe.get_time_transformation_env().pca_dev_info.resize(1);
    }

    pipe.get_time_transformation_env().gpu_time_transformation_queue->resize(new_value);

    pipe.get_compute_cache().virtual_synchronize_W<TimeTransformationCutsEnable>(pipe);

    request_pipe_refresh();
}

template <>
void ComputePipeRequestOnSync::operator()<Convolution>(const ConvolutionStruct& new_value, Pipe& pipe)
{
    LOG_UPDATE_ON_SYNC(Convolution);

    if (new_value.enabled == false)
        pipe.get_postprocess().dispose();
    else if (new_value.enabled == true)
        pipe.get_postprocess().init();

    pipe.get_view_cache().virtual_synchronize_W<ViewXY>(pipe);
    pipe.get_view_cache().virtual_synchronize_W<ViewXZ>(pipe);
    pipe.get_view_cache().virtual_synchronize_W<ViewYZ>(pipe);
    request_pipe_refresh();
}

template <>
void ComputePipeRequestOnSync::operator()<TimeTransformationCutsEnable>(bool new_value, Pipe& pipe)
{
    LOG_UPDATE_ON_SYNC(TimeTransformationCutsEnable);

    if (new_value == false)
    {

        pipe.get_buffers().gpu_postprocess_frame_xz.reset(nullptr);
        pipe.get_buffers().gpu_postprocess_frame_yz.reset(nullptr);
        pipe.get_buffers().gpu_output_frame_xz.reset(nullptr);
        pipe.get_buffers().gpu_output_frame_yz.reset(nullptr);

        pipe.get_time_transformation_env().gpu_output_queue_xz.reset(nullptr);
        pipe.get_time_transformation_env().gpu_output_queue_yz.reset(nullptr);
    }
    else
    {
        FrameDescriptor fd_xz = pipe.get_gpu_output_queue().get_fd();

        fd_xz.depth = sizeof(ushort);
        auto fd_yz = fd_xz;
        fd_xz.height = api::detail::get_value<TimeTransformationSize>();
        fd_yz.width = api::detail::get_value<TimeTransformationSize>();

        pipe.get_time_transformation_env().gpu_output_queue_xz.reset(
            new Queue(fd_xz, api::detail::get_value<TimeTransformationCutsBufferSize>()));
        pipe.get_time_transformation_env().gpu_output_queue_yz.reset(
            new Queue(fd_yz, api::detail::get_value<TimeTransformationCutsBufferSize>()));

        pipe.get_buffers().gpu_postprocess_frame_xz.resize(fd_xz.get_frame_res());
        pipe.get_buffers().gpu_postprocess_frame_yz.resize(fd_yz.get_frame_res());

        pipe.get_buffers().gpu_output_frame_xz.resize(fd_xz.get_frame_res());
        pipe.get_buffers().gpu_output_frame_yz.resize(fd_yz.get_frame_res());
    }

    pipe.get_view_cache().virtual_synchronize_W<ViewXZ>(pipe);
    pipe.get_view_cache().virtual_synchronize_W<ViewYZ>(pipe);

    request_pipe_refresh();
}

} // namespace holovibes
