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

bool ICompute::update_time_transformation_size(const unsigned short time_transformation_size) {
    try {
        resize_gpu_p_acc_buffer(time_transformation_size);
        perform_time_transformation_setting_specific_tasks(time_transformation_size);
        resize_gpu_time_transformation_queue(time_transformation_size);
    } catch (const std::exception& e) {
        handle_exception(e);
        return false;
    }
    return true;
}

void ICompute::resize_gpu_p_acc_buffer(const unsigned short time_transformation_size) {
    auto frame_res = gpu_input_queue_.get_fd().get_frame_res();
    time_transformation_env_.gpu_p_acc_buffer.resize(frame_res * time_transformation_size);
}

void ICompute::perform_time_transformation_setting_specific_tasks(const unsigned short time_transformation_size) {
    switch (setting<settings::TimeTransformation>()) {
        case TimeTransformation::STFT:
        case TimeTransformation::SSA_STFT:
            update_stft(time_transformation_size);
            if (setting<settings::TimeTransformation>() == TimeTransformation::SSA_STFT) {
                update_pca(time_transformation_size);
            }
            break;
        case TimeTransformation::PCA:
            update_pca(time_transformation_size);
            break;
        case TimeTransformation::NONE:
            break;
        default:
            LOG_ERROR("Unhandled Time transformation settings");
            break;
    }
}

void ICompute::update_stft(const unsigned short time_transformation_size) {
    int inembed_stft[1] = {time_transformation_size};
    int zone_size = static_cast<int>(gpu_input_queue_.get_fd().get_frame_res());
    time_transformation_env_.stft_plan.planMany(1, inembed_stft, inembed_stft, zone_size, 1, inembed_stft, zone_size, 1, CUFFT_C2C, zone_size);
}

void ICompute::update_pca(const unsigned short time_transformation_size) {
    auto size = static_cast<const uint>(time_transformation_size);
    time_transformation_env_.pca_cov.resize(size * size);
    time_transformation_env_.pca_eigen_values.resize(time_transformation_size);
    time_transformation_env_.pca_dev_info.resize(1);
}

void ICompute::resize_gpu_time_transformation_queue(const unsigned short time_transformation_size) {
    time_transformation_env_.gpu_time_transformation_queue->resize(time_transformation_size, stream_);
}

void ICompute::handle_exception(const std::exception& e) {
    time_transformation_env_.gpu_time_transformation_queue.reset(nullptr);
    request_time_transformation_cuts_ = false;
    request_delete_time_transformation_cuts_ = true;
    dispose_cuts();
    LOG_ERROR("error in update_time_transformation_size(time_transformation_size) message: {}", e.what());
}

void ICompute::update_spatial_transformation_parameters()
{
    const auto& gpu_input_queue_fd = gpu_input_queue_.get_fd();
    batch_env_.batch_index = 0;
    // We avoid the depth in the multiplication because the resize already take
    // it into account
    buffers_.gpu_spatial_transformation_buffer.resize(setting<settings::BatchSize>() *
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
        gpu_input_queue_fd.get_frame_res(), // Ouput layout same as input
        CUDA_C_32F,                         // Output type
        setting<settings::BatchSize>(),    // Batch size
        CUDA_C_32F);                        // Computation type
}

void ICompute::init_cuts()
{
   camera::FrameDescriptor fd_xz = gpu_output_queue_.get_fd();

    fd_xz.depth = sizeof(ushort);
    auto fd_yz = fd_xz;
    fd_xz.height = setting<settings::TimeTransformationSize>();
    fd_yz.width = setting<settings::TimeTransformationSize>();

    time_transformation_env_.gpu_output_queue_xz.reset(
        new Queue(fd_xz, setting<settings::TimeTransformationCutsOutputBufferSize>()));
    time_transformation_env_.gpu_output_queue_yz.reset(
        new Queue(fd_yz, setting<settings::TimeTransformationCutsOutputBufferSize>()));

    buffers_.gpu_postprocess_frame_xz.resize(fd_xz.get_frame_res());
    buffers_.gpu_postprocess_frame_yz.resize(fd_yz.get_frame_res());

    buffers_.gpu_output_frame_xz.resize(fd_xz.get_frame_res());
    buffers_.gpu_output_frame_yz.resize(fd_yz.get_frame_res());
}

void ICompute::dispose_cuts()
{
    buffers_.gpu_postprocess_frame_xz.reset(nullptr);
    buffers_.gpu_postprocess_frame_yz.reset(nullptr);
    /*
    buffers_.gpu_postprocess_frame_xz_final.reset(nullptr);
    buffers_.gpu_postprocess_frame_yz_final.reset(nullptr);
    */
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

std::unique_ptr<Queue>& ICompute::get_frame_record_queue() { return frame_record_env_.frame_record_queue_; }

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

void ICompute::request_frame_record()
{
    frame_record_requested_ = true;
    request_refresh();
}

void ICompute::request_disable_frame_record()
{
    disable_frame_record_requested_ = true;
    request_refresh();
}

void ICompute::request_autocontrast(WindowKind kind)
{
    if (kind == WindowKind::XYview && setting<settings::XY>().contrast.enabled){
        autocontrast_requested_ = true;
    }
    else if (kind == WindowKind::XZview && setting<settings::XZ>().contrast.enabled && setting<settings::CutsViewEnabled>())
        autocontrast_slice_xz_requested_ = true;
    else if (kind == WindowKind::YZview && setting<settings::CutsViewEnabled>())
        autocontrast_slice_yz_requested_ = true;
    else if (kind == WindowKind::Filter2D && setting<settings::Filter2d>().contrast.enabled &&
             setting<settings::Filter2dEnabled>())
        autocontrast_filter2d_requested_ = true;
}

void ICompute::request_update_time_transformation_size()
{
    update_time_transformation_size_requested_ = true;
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

void ICompute::request_update_time_stride()
{
    request_update_time_stride_ = true;
    request_refresh();
}

void ICompute::request_disable_lens_view()
{
    request_disable_lens_view_ = true;
    request_refresh();
}

void ICompute::request_clear_img_acc()
{
    request_clear_img_accu = true;
    request_refresh();
}

void ICompute::request_convolution()
{
    convolution_requested_ = true;
    request_refresh();
}

void ICompute::request_filter()
{
    filter_requested_ = true;
    request_refresh();
}

void ICompute::request_disable_convolution()
{
    disable_convolution_requested_ = true;
    request_refresh();
}

void ICompute::request_disable_filter()
{
    disable_filter_requested_ = true;
    request_refresh();
}
} // namespace holovibes
