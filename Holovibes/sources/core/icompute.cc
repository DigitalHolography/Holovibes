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
#include "pipe.hh"
#include "logger.hh"

#include "holovibes.hh"

namespace holovibes
{
using camera::FrameDescriptor;

ICompute::ICompute(BatchInputQueue& input, Queue& output, const cudaStream_t& stream)
    : gpu_input_queue_(input)
    , gpu_output_queue_(output)
    , stream_(stream)
    , past_time_(std::chrono::high_resolution_clock::now())
{
    int err = 0;

    plan_unwrap_2d_.plan(gpu_input_queue_.get_fd().width, gpu_input_queue_.get_fd().height, CUFFT_C2C);

    const camera::FrameDescriptor& fd = gpu_input_queue_.get_fd();
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
                                            CUDA_C_32F);                           // Computation type

    int inembed[1];
    int zone_size = static_cast<int>(gpu_input_queue_.get_fd().get_frame_res());

    inembed[0] = compute_cache_.get_value<TimeTransformationSize>();

    time_transformation_env_.stft_plan
        .planMany(1, inembed, inembed, zone_size, 1, inembed, zone_size, 1, CUFFT_C2C, zone_size);

    camera::FrameDescriptor new_fd = gpu_input_queue_.get_fd();
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
    if (view_cache_.get_value<ImgTypeParam>() == ImgType::Composite)
        image::grey_to_rgb_size(output_buffer_size);
    if (!buffers_.gpu_output_frame.resize(output_buffer_size))
        err++;
    buffers_.gpu_postprocess_frame_size = static_cast<int>(gpu_input_queue_.get_fd().get_frame_res());

    if (view_cache_.get_value<ImgTypeParam>() == ImgType::Composite)
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

bool ICompute::update_time_transformation_size(const unsigned short time_transformation_size)
{
    time_transformation_env_.gpu_p_acc_buffer.resize(gpu_input_queue_.get_fd().get_frame_res() *
                                                     time_transformation_size);

    if (compute_cache_.get_value<TimeTransformationParam>() == TimeTransformation::STFT)
    {
        /* CUFFT plan1d realloc */
        int inembed_stft[1] = {time_transformation_size};

        int zone_size = static_cast<int>(gpu_input_queue_.get_fd().get_frame_res());

        time_transformation_env_.stft_plan
            .planMany(1, inembed_stft, inembed_stft, zone_size, 1, inembed_stft, zone_size, 1, CUFFT_C2C, zone_size);
    }
    else if (compute_cache_.get_value<TimeTransformationParam>() == TimeTransformation::PCA)
    {
        // Pre allocate all the buffer only when n changes to avoid 1 allocation
        // every frame Static cast to avoid ushort overflow
        time_transformation_env_.pca_cov.resize(static_cast<const uint>(time_transformation_size) *
                                                time_transformation_size);
        time_transformation_env_.pca_eigen_values.resize(time_transformation_size);
        time_transformation_env_.pca_dev_info.resize(1);
    }
    else if (compute_cache_.get_value<TimeTransformationParam>() == TimeTransformation::NONE)
    {
        // Nothing to do
    }
    else if (compute_cache_.get_value<TimeTransformationParam>() == TimeTransformation::SSA_STFT)
    {
        /* CUFFT plan1d realloc */
        int inembed_stft[1] = {time_transformation_size};

        int zone_size = static_cast<int>(gpu_input_queue_.get_fd().get_frame_res());

        time_transformation_env_.stft_plan
            .planMany(1, inembed_stft, inembed_stft, zone_size, 1, inembed_stft, zone_size, 1, CUFFT_C2C, zone_size);

        // Pre allocate all the buffer only when n changes to avoid 1 allocation
        // every frame Static cast to avoid ushort overflow
        time_transformation_env_.pca_cov.resize(static_cast<const uint>(time_transformation_size) *
                                                time_transformation_size);
        time_transformation_env_.pca_eigen_values.resize(time_transformation_size);
        time_transformation_env_.pca_dev_info.resize(1);
    }
    else // Should not happend or be handled (if add more time transformation)
        CHECK(false);

    try
    {
        /* This will resize cuts buffers: Some modifications are to be applied
         * on opengl to work */
        time_transformation_env_.gpu_time_transformation_queue->resize(time_transformation_size, stream_);
    }
    catch (const std::exception& e)
    {
        time_transformation_env_.gpu_time_transformation_queue.reset(nullptr);

        request_time_transformation_cuts_ = false;

        request_delete_time_transformation_cuts_ = true;
        dispose_cuts();
        LOG_ERROR("error in update_time_transformation_size(time_transformation_size) message: {}", e.what());
        return false;
    }

    return true;
}

void ICompute::update_spatial_transformation_parameters()
{
    const auto& gpu_input_queue_fd = gpu_input_queue_.get_fd();
    batch_env_.batch_index = 0;
    // We avoid the depth in the multiplication because the resize already take
    // it into account
    buffers_.gpu_spatial_transformation_buffer.resize(compute_cache_.get_value<BatchSize>() *
                                                      gpu_input_queue_fd.get_frame_res());

    long long int n[] = {gpu_input_queue_fd.height, gpu_input_queue_fd.width};

    // This plan has a useful significant memory cost, check XtplanMany comment
    spatial_transformation_plan_.XtplanMany(
        2,                                  // 2D
        n,                                  // Dimension of inner most & outer most dimension
        n,                                  // Storage dimension size
        1,                                  // Between two inputs (pixels) of same image distance is one
        gpu_input_queue_fd.get_frame_res(), // Distance between 2 same index pixels of 2 images
        CUDA_C_32F,                         // Input type
        n,
        1,
        gpu_input_queue_fd.get_frame_res(),    // Ouput layout same as input
        CUDA_C_32F,                            // Output type
        compute_cache_.get_value<BatchSize>(), // Batch size
        CUDA_C_32F);                           // Computation type
}

void ICompute::init_cuts()
{
    camera::FrameDescriptor fd_xz = gpu_output_queue_.get_fd();

    fd_xz.depth = sizeof(ushort);
    auto fd_yz = fd_xz;
    fd_xz.height = GSH::instance().get_value<TimeTransformationSize>();
    fd_yz.width = GSH::instance().get_value<TimeTransformationSize>();

    time_transformation_env_.gpu_output_queue_xz.reset(
        new Queue(fd_xz, GSH::instance().get_value<TimeTransformationCutsOutputBufferSize>()));
    time_transformation_env_.gpu_output_queue_yz.reset(
        new Queue(fd_yz, GSH::instance().get_value<TimeTransformationCutsOutputBufferSize>()));

    buffers_.gpu_postprocess_frame_xz.resize(fd_xz.get_frame_res());
    buffers_.gpu_postprocess_frame_yz.resize(fd_yz.get_frame_res());

    buffers_.gpu_output_frame_xz.resize(fd_xz.get_frame_res());
    buffers_.gpu_output_frame_yz.resize(fd_yz.get_frame_res());
}

void ICompute::dispose_cuts()
{
    buffers_.gpu_postprocess_frame_xz.reset(nullptr);
    buffers_.gpu_postprocess_frame_yz.reset(nullptr);
    buffers_.gpu_output_frame_xz.reset(nullptr);
    buffers_.gpu_output_frame_yz.reset(nullptr);

    time_transformation_env_.gpu_output_queue_xz.reset(nullptr);
    time_transformation_env_.gpu_output_queue_yz.reset(nullptr);
}

std::unique_ptr<Queue>& ICompute::get_raw_view_queue() { return gpu_raw_view_queue_; }

std::unique_ptr<Queue>& ICompute::get_filter2d_view_queue() { return gpu_filter2d_view_queue_; }

std::unique_ptr<ConcurrentDeque<ChartPoint>>& ICompute::get_chart_display_queue()
{
    return chart_env_.chart_display_queue_;
}

std::unique_ptr<ConcurrentDeque<ChartPoint>>& ICompute::get_chart_record_queue()
{
    return chart_env_.chart_record_queue_;
}

std::unique_ptr<Queue>& ICompute::get_frame_record_queue() { return frame_record_env_.gpu_frame_record_queue_; }

bool ICompute::get_cuts_delete_request() { return request_delete_time_transformation_cuts_; }

std::unique_ptr<Queue>& ICompute::get_stft_slice_queue(int slice)
{
    return slice ? time_transformation_env_.gpu_output_queue_yz : time_transformation_env_.gpu_output_queue_xz;
}

/*
    FIXME: Need to delete because of merge ?
void ICompute::pipe_error(const int& err_count, const std::exception& e)
{
    LOG_ERROR("Pipe error: ");
    LOG_ERROR("  message: {}", e.what());
    LOG_ERROR("  err_count: {}", err_count);
    notify_error_observers(e);
}
*/

void ICompute::request_refresh() { refresh_requested_ = true; }

void ICompute::request_termination() { termination_requested_ = true; }

void ICompute::request_autocontrast(WindowKind kind)
{
    if (kind == WindowKind::XYview && view_cache_.get_value<ViewXY>().contrast.enabled)
        view_cache_.get_value<ViewXY>().request_exec_auto_contrast();
    else if (kind == WindowKind::XZview && view_cache_.get_value<ViewXZ>().contrast.enabled &&
             view_cache_.get_value<CutsViewEnabled>())
        view_cache_.get_value<ViewXZ>().request_exec_auto_contrast();
    else if (kind == WindowKind::YZview && view_cache_.get_value<ViewYZ>().contrast.enabled &&
             view_cache_.get_value<CutsViewEnabled>())
        view_cache_.get_value<ViewYZ>().request_exec_auto_contrast();
    else if (kind == WindowKind::Filter2D && view_cache_.get_value<Filter2D>().contrast.enabled &&
             view_cache_.get_value<Filter2DEnabled>())
        view_cache_.get_value<Filter2D>().request_exec_auto_contrast();
}
} // namespace holovibes
