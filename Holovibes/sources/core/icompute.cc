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

#include "tools_compute.cuh"
#include "compute_bundles.hh"
#include "update_exception.hh"
#include "unique_ptr.hh"
#include "pipe.hh"
#include "logger.hh"

#include "API.hh"

#include "holovibes.hh"

namespace holovibes
{
using camera::FrameDescriptor;

bool ICompute::update_time_transformation_size(const unsigned short size)
{
    try
    {
        // Updates the size of the GPU P acc buffer.
        auto frame_res = input_queue_.get_fd().get_frame_res();
        time_transformation_env_.gpu_p_acc_buffer.resize(frame_res * size);

        perform_time_transformation_setting_specific_tasks(size);

        // Resize the time transformation queue
        time_transformation_env_.gpu_time_transformation_queue->resize(size, stream_);
    }
    catch (const std::exception& e)
    {
        time_transformation_env_.gpu_time_transformation_queue.reset(nullptr);
        clear_request(ICS::TimeTransformationCuts);
        set_requested(ICS::DeleteTimeTransformationCuts, true);
        dispose_cuts();
        LOG_ERROR("error in update_time_transformation_size(time_transformation_size) message: {}", e.what());

        return false;
    }
    return true;
}

void ICompute::update_stft(const unsigned short size)
{
    int inembed_stft[1] = {size};
    int zone_size = static_cast<int>(input_queue_.get_fd().get_frame_res());
    time_transformation_env_.stft_plan
        .planMany(1, inembed_stft, inembed_stft, zone_size, 1, inembed_stft, zone_size, 1, CUFFT_C2C, zone_size);
}

void ICompute::update_pca(const unsigned short size)
{
    size_t st_size = size;
    time_transformation_env_.pca_cov.resize(st_size * st_size);
    time_transformation_env_.pca_eigen_values.resize(size);
    time_transformation_env_.pca_dev_info.resize(1);
}

void ICompute::perform_time_transformation_setting_specific_tasks(const unsigned short size)
{
    switch (setting<settings::TimeTransformation>())
    {
    case TimeTransformation::STFT:
        update_stft(size);
        break;
    case TimeTransformation::SSA_STFT:
        update_stft(size);
        update_pca(size);
        break;
    case TimeTransformation::PCA:
        update_pca(size);
        break;
    case TimeTransformation::NONE:
        break;
    default:
        LOG_ERROR("Unhandled Time transformation settings");
        break;
    }
}

void ICompute::update_spatial_transformation_parameters()
{
    const auto& input_queue_fd = input_queue_.get_fd();
    batch_env_.batch_index = 0;
    // We avoid the depth in the multiplication because the resize already take
    // it into account
    // buffers_.gpu_spatial_transformation_buffer.resize(setting<settings::BatchSize>() *
    // input_queue_fd.get_frame_res());
    buffers_.gpu_spatial_transformation_queue.get()->resize(setting<settings::BatchSize>(),
                                                            //    input_queue_fd.get_frame_res(),
                                                            stream_);

    long long int n[] = {input_queue_fd.height, input_queue_fd.width};

    // This plan has a useful significant memory cost, check XtplanMany comment
    spatial_transformation_plan_.XtplanMany(
        2,                              // 2D
        n,                              // Dimension of inner most & outer most dimension
        n,                              // Storage dimension size
        1,                              // Between two inputs (pixels) of same image distance is one
        input_queue_fd.get_frame_res(), // Distance between 2 same index pixels of 2 images
        CUDA_C_32F,                     // Input type
        n,
        1,
        input_queue_fd.get_frame_res(), // Ouput layout same as input
        CUDA_C_32F,                     // Output type
        setting<settings::BatchSize>(), // Batch size
        CUDA_C_32F);                    // Computation type
}

void ICompute::init_cuts()
{
    camera::FrameDescriptor fd_xz = gpu_output_queue_.get_fd();

    fd_xz.depth = camera::PixelDepth::Bits16; // Size of ushort
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

    image_acc_env_.gpu_accumulation_xz_queue.reset(nullptr);
    image_acc_env_.gpu_accumulation_yz_queue.reset(nullptr);

    buffers_.gpu_output_frame_xz.reset(nullptr);
    buffers_.gpu_output_frame_yz.reset(nullptr);

    time_transformation_env_.gpu_output_queue_xz.reset(nullptr);
    time_transformation_env_.gpu_output_queue_yz.reset(nullptr);
}

void ICompute::request_autocontrast(WindowKind kind)
{
    if (kind == WindowKind::XYview && setting<settings::XY>().contrast.enabled)
        set_requested(ICS::Autocontrast, true);
    else if (kind == WindowKind::XZview && setting<settings::XZ>().contrast.enabled &&
             setting<settings::CutsViewEnabled>())
        set_requested(ICS::AutocontrastSliceXZ, true);
    else if (kind == WindowKind::YZview && setting<settings::YZ>().contrast.enabled &&
             setting<settings::CutsViewEnabled>())
        set_requested(ICS::AutocontrastSliceYZ, true);
    else if (kind == WindowKind::Filter2D && setting<settings::Filter2d>().contrast.enabled &&
             setting<settings::Filter2dEnabled>())
        set_requested(ICS::AutocontrastFilter2D, true);
}

void ICompute::request_record_chart(unsigned int nb_chart_points_to_record)
{
    chart_record_requested_ = nb_chart_points_to_record;
    request_refresh();
}

void ICompute::request_refresh() { set_requested(ICS::Refresh, true); }

// Start
std::atomic<bool>& ICompute::is_requested(Setting setting) { return settings_requests_[static_cast<int>(setting)]; }

void ICompute::request(Setting setting)
{
    settings_requests_[static_cast<int>(setting)] = true;
    request_refresh();
}

void ICompute::set_requested(Setting setting, bool value) { settings_requests_[static_cast<int>(setting)] = value; }

void ICompute::clear_request(Setting setting) { settings_requests_[static_cast<int>(setting)] = false; }
} // namespace holovibes
