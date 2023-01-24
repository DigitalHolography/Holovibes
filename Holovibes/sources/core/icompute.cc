#include <cassert>

#include "icompute.hh"
#include "fft1.cuh"
#include "fft2.cuh"
#include "stft.cuh"
#include "tools.cuh"
#include "contrast_correction.cuh"
#include "chart.cuh"
#include "queue.hh"
#include "concurrent_deque.hh"

#include "power_of_two.hh"
#include "tools_compute.cuh"
#include "compute_bundles.hh"
#include "update_exception.hh"
#include "unique_ptr.hh"
#include "logger.hh"
#include "API.hh"

#include "holovibes.hh"

namespace holovibes
{

ICompute::ICompute(BatchInputQueue& input, Queue& output, const cudaStream_t& stream)
    : gpu_input_queue_(input)
    , gpu_output_queue_(output)
    , stream_(stream)
    , past_time_(std::chrono::high_resolution_clock::now())
{
    int err = 0;

    plan_unwrap_2d_.plan(gpu_input_queue_.get_fd().width, gpu_input_queue_.get_fd().height, CUFFT_C2C);

    const FrameDescriptor& fd = gpu_input_queue_.get_fd();
    long long int n[] = {fd.height, fd.width};

    // This plan has a useful significant memory cost, check XtplanMany comment
    spatial_transformation_plan_.XtplanMany(2, // 2D
                                            n, // Dimension of inner most & outer most dimension
                                            n, // Storage dimension size
                                            1, // Between two inputs (pixels) of same image distance is one
                                            fd.get_frame_res(), // Distance between 2 same index pixels of 2 images
                                            CUDA_C_32F,         // Input type
                                            n,
                                            1,
                                            fd.get_frame_res(),                    // Ouput layout same as input
                                            CUDA_C_32F,                            // Output type
                                            compute_cache_.get_value<BatchSize>(), // Batch size
                                            CUDA_C_32F);                           // ComputeModeEnum type

    int inembed[1];
    int zone_size = static_cast<int>(gpu_input_queue_.get_fd().get_frame_res());

    inembed[0] = compute_cache_.get_value<TimeTransformationSize>();

    time_transformation_env_.stft_plan
        .planMany(1, inembed, inembed, zone_size, 1, inembed, zone_size, 1, CUFFT_C2C, zone_size);

    FrameDescriptor new_fd = gpu_input_queue_.get_fd();
    new_fd.depth = 8;
    // FIXME-CAMERA : WTF depth 8 ==> maybe a magic value for complex mode
    time_transformation_env_.gpu_time_transformation_queue.reset(
        new Queue(new_fd, compute_cache_.get_value<TimeTransformationSize>()));

    // Static cast size_t to avoid overflow
    if (!buffers_.gpu_spatial_transformation_buffer.resize(
            static_cast<const size_t>(compute_cache_.get_value<BatchSize>()) *
            gpu_input_queue_.get_fd().get_frame_res()))
        err++;

    int output_buffer_size = gpu_input_queue_.get_fd().get_frame_res();
    if (compute_cache_.get_value<ImageType>() == ImageTypeEnum::Composite)
        image::grey_to_rgb_size(output_buffer_size);
    if (!buffers_.gpu_output_frame.resize(output_buffer_size))
        err++;
    buffers_.gpu_postprocess_frame_size = static_cast<int>(gpu_input_queue_.get_fd().get_frame_res());

    if (compute_cache_.get_value<ImageType>() == ImageTypeEnum::Composite)
        image::grey_to_rgb_size(buffers_.gpu_postprocess_frame_size);

    if (!buffers_.gpu_postprocess_frame.resize(buffers_.gpu_postprocess_frame_size))
        err++;

    // Init the gpu_p_frame with the size of input image
    if (!time_transformation_env_.gpu_p_frame.resize(buffers_.gpu_postprocess_frame_size))
        err++;

    if (!buffers_.gpu_complex_filter2d_frame.resize(buffers_.gpu_postprocess_frame_size))
        err++;

    if (!buffers_.gpu_float_filter2d_frame.resize(buffers_.gpu_postprocess_frame_size))
        err++;

    if (!buffers_.gpu_filter2d_frame.resize(buffers_.gpu_postprocess_frame_size))
        err++;

    if (!buffers_.gpu_filter2d_mask.resize(output_buffer_size))
        err++;

    if (err != 0)
        throw std::exception(cudaGetErrorString(cudaGetLastError()));
}

ICompute::~ICompute() {}

std::unique_ptr<Queue>& ICompute::get_stft_slice_queue(int slice)
{
    return slice ? time_transformation_env_.gpu_output_queue_yz : time_transformation_env_.gpu_output_queue_xz;
}

} // namespace holovibes
