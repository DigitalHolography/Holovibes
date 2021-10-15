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
#include "compute_descriptor.hh"
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

ICompute::ICompute(BatchInputQueue& input, Queue& output, ComputeDescriptor& cd, const cudaStream_t& stream)
    : cd_(cd)
    , gpu_input_queue_(input)
    , gpu_output_queue_(output)
    , stream_(stream)
    , past_time_(std::chrono::high_resolution_clock::now())
{
    int err = 0;

    plan_unwrap_2d_.plan(gpu_input_queue_.get_fd().width, gpu_input_queue_.get_fd().height, CUFFT_C2C);

    const camera::FrameDescriptor& fd = gpu_input_queue_.get_fd();
    long long int n[] = {fd.height, fd.width};

    // This plan has a useful significant memory cost, check XtplanMany comment
    spatial_transformation_plan_.XtplanMany(2,              // 2D
                                            n,              // Dimension of inner most & outer most dimension
                                            n,              // Storage dimension size
                                            1,              // Between two inputs (pixels) of same image distance is one
                                            fd.frame_res(), // Distance between 2 same index pixels of 2 images
                                            CUDA_C_32F,     // Input type
                                            n,
                                            1,
                                            fd.frame_res(), // Ouput layout same as input
                                            CUDA_C_32F,     // Output type
                                            cd_.batch_size, // Batch size
                                            CUDA_C_32F);    // Computation type

    int inembed[1];
    int zone_size = gpu_input_queue_.get_frame_res();

    inembed[0] = cd_.time_transformation_size;

    time_transformation_env_.stft_plan
        .planMany(1, inembed, inembed, zone_size, 1, inembed, zone_size, 1, CUFFT_C2C, zone_size);

    camera::FrameDescriptor new_fd = gpu_input_queue_.get_fd();
    new_fd.depth = 8;
    time_transformation_env_.gpu_time_transformation_queue.reset(new Queue(new_fd, cd_.time_transformation_size));

    // Static cast size_t to avoid overflow
    if (!buffers_.gpu_spatial_transformation_buffer.resize(static_cast<const size_t>(cd_.batch_size) *
                                                           gpu_input_queue_.get_fd().frame_res()))
        err++;

    int output_buffer_size = gpu_input_queue_.get_frame_res();
    if (cd_.img_type == ImgType::Composite)
        image::grey_to_rgb_size(output_buffer_size);
    if (!buffers_.gpu_output_frame.resize(output_buffer_size))
        err++;
    buffers_.gpu_postprocess_frame_size = gpu_input_queue_.get_frame_res();

    if (cd_.img_type == ImgType::Composite)
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

bool ICompute::update_time_transformation_size(const unsigned short time_transformation_size)
{
    unsigned int err_count = 0;
    time_transformation_env_.gpu_p_acc_buffer.resize(gpu_input_queue_.get_frame_res() * time_transformation_size);

    if (cd_.time_transformation == TimeTransformation::STFT)
    {
        /* CUFFT plan1d realloc */
        int inembed_stft[1] = {time_transformation_size};

        int zone_size = gpu_input_queue_.get_frame_res();

        time_transformation_env_.stft_plan
            .planMany(1, inembed_stft, inembed_stft, zone_size, 1, inembed_stft, zone_size, 1, CUFFT_C2C, zone_size);
    }
    else if (cd_.time_transformation == TimeTransformation::PCA)
    {
        // Pre allocate all the buffer only when n changes to avoid 1 allocation
        // every frame Static cast to avoid ushort overflow
        time_transformation_env_.pca_cov.resize(static_cast<const uint>(time_transformation_size) *
                                                time_transformation_size);
        time_transformation_env_.pca_eigen_values.resize(time_transformation_size);
        time_transformation_env_.pca_dev_info.resize(1);
    }
    else if (cd_.time_transformation == TimeTransformation::NONE)
    {
        // Nothing to do
    }
    else if (cd_.time_transformation == TimeTransformation::SSA_STFT)
    {
        /* CUFFT plan1d realloc */
        int inembed_stft[1] = {time_transformation_size};

        int zone_size = gpu_input_queue_.get_frame_res();

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
    catch (std::exception&)
    {
        time_transformation_env_.gpu_time_transformation_queue.reset(nullptr);
        request_time_transformation_cuts_ = false;
        request_delete_time_transformation_cuts_ = true;
        dispose_cuts();
        err_count++;
    }

    if (err_count != 0)
    {
        pipe_error(err_count, UpdateException("error in update_time_transformation_size(time_transformation_size)"));
        return false;
    }

    notify_observers();
    return true;
}

void ICompute::update_spatial_transformation_parameters()
{
    const auto& gpu_input_queue_fd = gpu_input_queue_.get_fd();
    batch_env_.batch_index = 0;
    // We avoid the depth in the multiplication because the resize already take
    // it into account
    buffers_.gpu_spatial_transformation_buffer.resize(cd_.batch_size * gpu_input_queue_fd.frame_res());

    long long int n[] = {gpu_input_queue_fd.height, gpu_input_queue_fd.width};

    // This plan has a useful significant memory cost, check XtplanMany comment
    spatial_transformation_plan_.XtplanMany(
        2,                              // 2D
        n,                              // Dimension of inner most & outer most dimension
        n,                              // Storage dimension size
        1,                              // Between two inputs (pixels) of same image distance is one
        gpu_input_queue_fd.frame_res(), // Distance between 2 same index pixels of 2 images
        CUDA_C_32F,                     // Input type
        n,
        1,
        gpu_input_queue_fd.frame_res(), // Ouput layout same as input
        CUDA_C_32F,                     // Output type
        cd_.batch_size,                 // Batch size
        CUDA_C_32F);                    // Computation type
}

void ICompute::init_cuts()
{
    camera::FrameDescriptor fd_xz = gpu_output_queue_.get_fd();

    fd_xz.depth = sizeof(ushort);
    uint buffer_depth = sizeof(float);
    auto fd_yz = fd_xz;
    fd_xz.height = cd_.time_transformation_size;
    fd_yz.width = cd_.time_transformation_size;
    time_transformation_env_.gpu_output_queue_xz.reset(
        new Queue(fd_xz, cd_.time_transformation_cuts_output_buffer_size));
    time_transformation_env_.gpu_output_queue_yz.reset(
        new Queue(fd_yz, cd_.time_transformation_cuts_output_buffer_size));
    buffers_.gpu_postprocess_frame_xz.resize(fd_xz.frame_res());
    buffers_.gpu_postprocess_frame_yz.resize(fd_yz.frame_res());

    buffers_.gpu_output_frame_xz.resize(fd_xz.frame_res());
    buffers_.gpu_output_frame_yz.resize(fd_yz.frame_res());
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

void ICompute::delete_stft_slice_queue()
{
    request_delete_time_transformation_cuts_ = true;
    request_refresh();
}

void ICompute::create_stft_slice_queue()
{
    request_time_transformation_cuts_ = true;
    request_refresh();
}

bool ICompute::get_cuts_request() { return request_time_transformation_cuts_; }

bool ICompute::get_cuts_delete_request() { return request_delete_time_transformation_cuts_; }

std::unique_ptr<Queue>& ICompute::get_stft_slice_queue(int slice)
{
    return slice ? time_transformation_env_.gpu_output_queue_yz : time_transformation_env_.gpu_output_queue_xz;
}

void ICompute::pipe_error(const int& err_count, const std::exception& e)
{
    LOG_ERROR << "Pipe error: ";
    LOG_ERROR << "  message: " << e.what();
    LOG_ERROR << "  err_count: " << err_count;
    notify_error_observers(e);
}

void ICompute::soft_request_refresh()
{
    if (!refresh_requested_)
        refresh_requested_ = true;
}

void ICompute::request_refresh() { refresh_requested_ = true; }

void ICompute::request_termination() { termination_requested_ = true; }

void ICompute::request_output_resize(unsigned int new_output_size)
{
    output_resize_requested_ = new_output_size;
    request_refresh();
}

void ICompute::request_disable_raw_view()
{
    disable_raw_view_requested_ = true;
    request_refresh();
}

void ICompute::request_raw_view()
{
    raw_view_requested_ = true;
    request_refresh();
}

void ICompute::request_disable_filter2d_view()
{
    disable_filter2d_view_requested_ = true;
    request_refresh();
}

void ICompute::request_filter2d_view()
{
    filter2d_view_requested_ = true;
    request_refresh();
}

void ICompute::request_hologram_record(std::optional<unsigned int> nb_frames_to_record)
{
    hologram_record_requested_ = nb_frames_to_record;
    request_refresh();
}

void ICompute::request_raw_record(std::optional<unsigned int> nb_frames_to_record)
{
    raw_record_requested_ = nb_frames_to_record;
    request_refresh();
}

void ICompute::request_disable_frame_record()
{
    disable_frame_record_requested_ = true;
    request_refresh();
}

void ICompute::request_autocontrast(WindowKind kind)
{
    // Do not request anything if contrast is not enabled
    if (!cd_.xy.contrast_enabled)
        return;

    if (kind == WindowKind::XYview)
        autocontrast_requested_ = true;
    else if (kind == WindowKind::XZview)
        autocontrast_slice_xz_requested_ = true;
    else if (kind == WindowKind::YZview)
        autocontrast_slice_yz_requested_ = true;
    else if (kind == WindowKind::Filter2D)
        autocontrast_filter2d_requested_ = true;
}

void ICompute::request_update_time_transformation_size()
{
    update_time_transformation_size_requested_ = true;
    request_refresh();
}

void ICompute::request_update_unwrap_size(const unsigned size)
{
    cd_.unwrap_history_size = size;
    request_refresh();
}

void ICompute::request_unwrapping_1d(const bool value) { unwrap_1d_requested_ = value; }

void ICompute::request_unwrapping_2d(const bool value) { unwrap_2d_requested_ = value; }

void ICompute::request_display_chart()
{
    chart_display_requested_ = true;
    request_refresh();
}

void ICompute::request_disable_display_chart()
{
    disable_chart_display_requested_ = true;
    request_refresh();
}

void ICompute::request_record_chart(unsigned int nb_chart_points_to_record)
{
    chart_record_requested_ = nb_chart_points_to_record;
    request_refresh();
}

void ICompute::request_disable_record_chart()
{
    disable_chart_record_requested_ = true;
    request_refresh();
}

void ICompute::request_update_batch_size()
{
    request_update_batch_size_ = true;
    request_refresh();
}

void ICompute::request_update_time_transformation_stride()
{
    request_update_time_transformation_stride_ = true;
    request_refresh();
}

void ICompute::request_disable_lens_view()
{
    request_disable_lens_view_ = true;
    request_refresh();
}

void ICompute::request_clear_img_acc()
{
    request_clear_img_acc_ = true;
    request_refresh();
}

void ICompute::request_convolution()
{
    convolution_requested_ = true;
    request_refresh();
}

void ICompute::request_disable_convolution()
{
    disable_convolution_requested_ = true;
    request_refresh();
}
} // namespace holovibes
