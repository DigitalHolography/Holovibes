#include <algorithm>
#include <cmath>
#include <sstream>
#include <vector>

#include "fourier_transform.hh"

#include "cublas_handle.hh"
#include "cusolver_handle.hh"
#include "icompute.hh"

#include "tools_conversion.cuh"
#include "tools_compute.cuh"
#include "tools_unwrap.cuh"
#include "filter2D.cuh"
#include "input_filter.cuh"
#include "fresnel_transform.cuh"
#include "angular_spectrum.cuh"
#include "masks.cuh"
#include "stft.cuh"
#include "frame_reshape.cuh"
#include "cuda_tools/cufft_handle.hh"
#include "cuda_memory.cuh"
#include "queue.hh"
#include "shift_corners.cuh"
#include "apply_mask.cuh"
#include "matrix_operations.hh"
#include "delete_twin_image.cuh"
#include "delete_twin_image_masks.cuh"
#include "compute_bundles_2d.hh"
#include "logger.hh"

using holovibes::FunctionVector;
using holovibes::Queue;
using holovibes::compute::FourierTransform;

namespace
{
constexpr float kDeleteTwinPhaseGaussianSigma = 60.0f;

struct RectBounds
{
    int x_min;
    int x_max;
    int y_min;
    int y_max;
};

RectBounds sanitize_rect(const holovibes::units::RectFd& rect, int width, int height)
{
    int x1 = std::clamp(rect.x(), 0, width);
    int x2 = std::clamp(rect.right(), 0, width);
    int y1 = std::clamp(rect.y(), 0, height);
    int y2 = std::clamp(rect.bottom(), 0, height);

    if (x1 > x2)
        std::swap(x1, x2);
    if (y1 > y2)
        std::swap(y1, y2);

    return RectBounds{x1, x2, y1, y2};
}

RectBounds build_symmetric_rect(const RectBounds& rect, int width, int height)
{
    int x_min = std::clamp(width - rect.x_max, 0, width);
    int x_max = std::clamp(width - rect.x_min, 0, width);
    int y_min = std::clamp(height - rect.y_max, 0, height);
    int y_max = std::clamp(height - rect.y_min, 0, height);

    if (x_min > x_max)
        std::swap(x_min, x_max);
    if (y_min > y_max)
        std::swap(y_min, y_max);

    return RectBounds{x_min, x_max, y_min, y_max};
}
} // namespace

void FourierTransform::ensure_delete_twin_image_resources(size_t total_elements)
{
    if (total_elements == 0)
        return;

    if (delete_twin_elements_capacity_ < total_elements)
    {
        gpu_delete_twin_frequency_buffer_.resize(total_elements);
        gpu_delete_twin_phase_buffer_.resize(total_elements);
        gpu_delete_twin_phase_blurred_buffer_.resize(total_elements);
        gpu_delete_twin_gaussian_temp_buffer_.resize(total_elements);
        gpu_delete_twin_amplitude_buffer_.resize(total_elements);
        delete_twin_elements_capacity_ = total_elements;
    }

    const size_t frame_res = fd_.get_frame_res();
    if (!delete_twin_unwrap_res_)
        delete_twin_unwrap_res_ = std::make_unique<UnwrappingResources_2d>(frame_res, stream_);
    else if (delete_twin_unwrap_res_->image_resolution_ != frame_res)
        delete_twin_unwrap_res_->reallocate(frame_res);
}

void FourierTransform::prepare_delete_twin_gaussian_kernel(float sigma)
{
    if (sigma <= 0.0f)
        return;

    delete_twin_gaussian_radius_ = static_cast<int>(std::ceil(3.0f * sigma));
    const int kernel_size = 2 * delete_twin_gaussian_radius_ + 1;
    std::vector<float> host_kernel(kernel_size);
    const float denom = 2.0f * sigma * sigma;
    float sum = 0.0f;
    for (int i = -delete_twin_gaussian_radius_; i <= delete_twin_gaussian_radius_; ++i)
    {
        const float value = std::exp(-(static_cast<float>(i * i)) / denom);
        const int idx = i + delete_twin_gaussian_radius_;
        host_kernel[idx] = value;
        sum += value;
    }
    for (float& value : host_kernel)
        value /= sum;

    if (gpu_delete_twin_gaussian_kernel_.get_size() < static_cast<size_t>(kernel_size) * sizeof(float))
        gpu_delete_twin_gaussian_kernel_.resize(kernel_size);

    cudaXMemcpyAsync(gpu_delete_twin_gaussian_kernel_.get(),
                     host_kernel.data(),
                     kernel_size * sizeof(float),
                     cudaMemcpyHostToDevice,
                     stream_);
}

void FourierTransform::insert_fft(const uint width, const uint height)
{
    LOG_FUNC();

    auto space_transformation = setting<settings::SpaceTransformation>();
    bool filter2d_enabled = setting<settings::Filter2dEnabled>();
    if (filter2d_enabled)
    {
        update_filter2d_circles_mask(buffers_.gpu_filter2d_mask,
                                     width,
                                     height,
                                     setting<settings::Filter2dN1>(),
                                     setting<settings::Filter2dN2>(),
                                     setting<settings::Filter2dSmoothLow>(),
                                     setting<settings::Filter2dSmoothHigh>(),
                                     stream_);

        if (!setting<settings::InputFilter>().empty())
        {
            apply_filter(buffers_.gpu_filter2d_mask,
                         buffers_.gpu_input_filter_mask,
                         setting<settings::InputFilter>().data(),
                         width,
                         height,
                         stream_);
        }

        // In ANGULARSP we do an optimisation to compute the filter2d in the same
        // reciprocal space to reduce the number of fft calculation
        if (space_transformation != SpaceTransformation::ANGULARSP)
            insert_filter2d();
    }

    if (space_transformation == SpaceTransformation::NONE)
        return;

    bool should_enqueue_lens = false;

    switch (space_transformation)
    {
    case SpaceTransformation::FRESNELTR:
        insert_fresnel_transform();
        should_enqueue_lens = true;
        break;
    case SpaceTransformation::ANGULARSP:
        insert_angular_spectrum(filter2d_enabled);
        should_enqueue_lens = true;
        break;
    case SpaceTransformation::DELETE_TWIN_IMAGE:
        insert_delete_twin_image_transform();
        should_enqueue_lens = true;
        break;
    default:
        LOG_WARN("Unknown space transformation requested");
        return;
    }

    if (should_enqueue_lens)
        fn_compute_vect_->push_back([=]() { enqueue_lens(space_transformation); });
}

void FourierTransform::insert_filter2d()
{
    LOG_FUNC();

    fn_compute_vect_->push_back(
        [=]()
        {
            filter2D(buffers_.gpu_spatial_transformation_buffer,
                     buffers_.gpu_filter2d_mask,
                     buffers_.gpu_complex_filter2d_frame,
                     setting<settings::Filter2dEnabled>(),
                     setting<settings::BatchSize>(),
                     spatial_transformation_plan_,
                     fd_.width,
                     fd_.height,
                     stream_);
        });
}

void FourierTransform::insert_fresnel_transform()
{
    LOG_FUNC();

    fresnel_transform_lens(gpu_lens_.get(),
                           lens_side_size_,
                           fd_.height,
                           fd_.width,
                           setting<settings::Lambda>(),
                           setting<settings::ZDistance>(),
                           setting<settings::PixelSize>(),
                           stream_);

    void* input_output = buffers_.gpu_spatial_transformation_buffer.get();

    fn_compute_vect_->push_back(
        [=]()
        {
            fresnel_transform(static_cast<cuComplex*>(input_output),
                              static_cast<cuComplex*>(input_output),
                              setting<settings::BatchSize>(),
                              gpu_lens_.get(),
                              spatial_transformation_plan_,
                              fd_.get_frame_res(),
                              stream_);
        });
}

void FourierTransform::insert_angular_spectrum(bool filter2d_enabled)
{
    LOG_FUNC();

    angular_spectrum_lens(gpu_lens_.get(),
                          fd_.width,
                          fd_.height,
                          setting<settings::ZDistance>(),
                          setting<settings::Lambda>(),
                          setting<settings::PixelSize>() * 1e-6f,
                          setting<settings::PixelSize>() * 1e-6f,
                          stream_);

    shift_corners(gpu_lens_.get(), 1, fd_.width, fd_.height, stream_);

    if (filter2d_enabled)
        apply_mask(gpu_lens_.get(), buffers_.gpu_filter2d_mask.get(), fd_.width * fd_.height, 1, stream_);

    void* input_output = buffers_.gpu_spatial_transformation_buffer.get();

    fn_compute_vect_->push_back(
        [=]()
        {
            angular_spectrum(static_cast<cuComplex*>(input_output),
                             static_cast<cuComplex*>(input_output),
                             setting<settings::BatchSize>(),
                             gpu_lens_.get(),
                             buffers_.gpu_complex_filter2d_frame,
                             filter2d_enabled,
                             spatial_transformation_plan_,
                             fd_,
                             stream_);
        });
}

void FourierTransform::insert_delete_twin_image_transform()
{
    LOG_FUNC();

    const int width = fd_.width;
    const int height = fd_.height;
    const size_t frame_res = fd_.get_frame_res();
    const uint batch_size = setting<settings::BatchSize>();
    const size_t total_elements = frame_res * static_cast<size_t>(batch_size);
    ensure_delete_twin_image_resources(total_elements);
    prepare_delete_twin_gaussian_kernel(kDeleteTwinPhaseGaussianSigma);

    angular_spectrum_lens(gpu_lens_.get(),
                          width,
                          height,
                          setting<settings::ZDistance>(),
                          setting<settings::Lambda>(),
                          setting<settings::PixelSize>() * 1e-6f,
                          setting<settings::PixelSize>() * 1e-6f,
                          stream_);

    shift_corners(gpu_lens_.get(), 1, width, height, stream_);

    if (setting<settings::Filter2dEnabled>())
        apply_mask(gpu_lens_.get(), buffers_.gpu_filter2d_mask.get(), width * height, 1, stream_);

    fn_compute_vect_->push_back(
        [=]()
        {
            auto rect = setting<settings::DeleteTwinImageRectangle>();
            RectBounds sanitized = sanitize_rect(rect, width, height);
            RectBounds symmetric = build_symmetric_rect(sanitized, width, height);

            holovibes::cuda::build_delete_twin_image_masks(buffers_.gpu_delete_twin_image_mp_mask.get(),
                                                           buffers_.gpu_delete_twin_image_ma_mask.get(),
                                                           width,
                                                           height,
                                                           sanitized.x_min,
                                                           sanitized.x_max,
                                                           sanitized.y_min,
                                                           sanitized.y_max,
                                                           symmetric.x_min,
                                                           symmetric.x_max,
                                                           symmetric.y_min,
                                                           symmetric.y_max,
                                                           stream_);

            const int rect_width = sanitized.x_max - sanitized.x_min;
            const int rect_height = sanitized.y_max - sanitized.y_min;
            const int symmetric_width = symmetric.x_max - symmetric.x_min;
            const int symmetric_height = symmetric.y_max - symmetric.y_min;

            LOG_INFO("Delete twin masks rebuilt. Primary rect x:[{}:{}), y:[{}:{}), size:{}x{} px. "
                     "Symmetric rect x:[{}:{}), y:[{}:{}), size:{}x{} px.",
                     sanitized.x_min,
                     sanitized.x_max,
                     sanitized.y_min,
                     sanitized.y_max,
                     rect_width,
                     rect_height,
                     symmetric.x_min,
                     symmetric.x_max,
                     symmetric.y_min,
                     symmetric.y_max,
                     symmetric_width,
                     symmetric_height);
        });

    fn_compute_vect_->push_back(
        [=]()
        {
            const uint batch_size = setting<settings::BatchSize>();
            if (batch_size == 0)
                return;

            const size_t frame_res = fd_.get_frame_res();
            const size_t total_elements = frame_res * static_cast<size_t>(batch_size);
            ensure_delete_twin_image_resources(total_elements);

            cuComplex* spatial = static_cast<cuComplex*>(buffers_.gpu_spatial_transformation_buffer.get());
            cuComplex* freq_copy = gpu_delete_twin_frequency_buffer_.get();
            float* phase_buffer = gpu_delete_twin_phase_buffer_.get();
            float* phase_blurred = gpu_delete_twin_phase_blurred_buffer_.get();
            float* gaussian_temp = gpu_delete_twin_gaussian_temp_buffer_.get();
            float* amplitude = gpu_delete_twin_amplitude_buffer_.get();

            cufftSafeCall(cufftXtExec(spatial_transformation_plan_, spatial, spatial, CUFFT_FORWARD));

            cudaXMemcpyAsync(freq_copy, spatial, total_elements * sizeof(cuComplex), cudaMemcpyDeviceToDevice, stream_);

            apply_mask(spatial, buffers_.gpu_delete_twin_image_mp_mask.get(), frame_res, batch_size, stream_);
            shift_corners(spatial, batch_size, fd_.width, fd_.height, stream_);
            cufftSafeCall(cufftXtExec(spatial_transformation_plan_, spatial, spatial, CUFFT_INVERSE));
            complex_divide(spatial, frame_res, static_cast<float>(frame_res), batch_size, stream_);

            apply_mask(freq_copy, buffers_.gpu_delete_twin_image_ma_mask.get(), frame_res, batch_size, stream_);
            cufftSafeCall(cufftXtExec(spatial_transformation_plan_, freq_copy, freq_copy, CUFFT_INVERSE));
            complex_divide(freq_copy, frame_res, static_cast<float>(frame_res), batch_size, stream_);

            for (uint batch_idx = 0; batch_idx < batch_size; ++batch_idx)
            {
                cuComplex* ma_frame = freq_copy + batch_idx * frame_res;
                float* amplitude_frame = amplitude + batch_idx * frame_res;
                complex_to_modulus_oct(amplitude_frame, ma_frame, frame_res, 0, 0, stream_);

                cuComplex* mp_frame = spatial + batch_idx * frame_res;
                float* phase_frame = phase_buffer + batch_idx * frame_res;
                complex_to_argument(phase_frame, mp_frame, 0, 0, frame_res, stream_);
                unwrap_2d(phase_frame,
                          phase_frame,
                          delete_twin_plan_unwrap_2d_,
                          delete_twin_unwrap_res_.get(),
                          fd_,
                          stream_);
            }

            if (delete_twin_gaussian_radius_ > 0 && gpu_delete_twin_gaussian_kernel_.get())
            {
                gaussian_blur_batch(phase_buffer,
                                    gaussian_temp,
                                    phase_blurred,
                                    gpu_delete_twin_gaussian_kernel_.get(),
                                    delete_twin_gaussian_radius_,
                                    fd_.width,
                                    fd_.height,
                                    batch_size,
                                    stream_);
                subtract_arrays(phase_buffer, phase_buffer, phase_blurred, total_elements, stream_);
            }

            combine_amplitude_phase(spatial, amplitude, phase_buffer, total_elements, true, stream_);

            angular_spectrum(spatial,
                             spatial,
                             batch_size,
                             gpu_lens_.get(),
                             buffers_.gpu_complex_filter2d_frame,
                             false,
                             spatial_transformation_plan_,
                             fd_,
                             stream_);
        });
}

void FourierTransform::init_lens_queue()
{
    LOG_FUNC();

    if (!gpu_lens_queue_)
    {
        auto fd = fd_;
        fd.depth = camera::PixelDepth::Complex;
        gpu_lens_queue_ = std::make_unique<Queue>(fd, 16);
    }
}

std::unique_ptr<Queue>& FourierTransform::get_lens_queue()
{
    LOG_FUNC();

    return gpu_lens_queue_;
}

// Inserted
void FourierTransform::enqueue_lens(SpaceTransformation space_transformation)
{
    // LOG-USELESS LOG_FUNC();

    if (setting<settings::LensViewEnabled>())
    {
        // Getting the pointer in the location of the next enqueued element
        cuComplex* copied_lens_ptr = static_cast<cuComplex*>(gpu_lens_queue_->get_end());
        gpu_lens_queue_->enqueue(gpu_lens_, stream_);

        // For optimisation purposes, when ANGULARSP is activated, lens is shifted
        // We have to shift it again to ensure a good display
        if (space_transformation == SpaceTransformation::ANGULARSP)
            shift_corners(copied_lens_ptr, 1, fd_.width, fd_.height, stream_);
        // Normalizing the newly enqueued element
        normalize_complex(copied_lens_ptr, fd_.get_frame_res(), stream_);
    }
}

void FourierTransform::insert_time_transform()
{
    LOG_FUNC();

    auto time_transformation = setting<settings::TimeTransformation>();
    auto time_transformation_size = setting<settings::TimeTransformationSize>();

    switch (time_transformation)
    {
    case TimeTransformation::STFT:
        insert_stft();
        break;
    case TimeTransformation::PCA:
        insert_pca();
        break;
    case TimeTransformation::SSA_STFT:
        insert_ssa_stft();
        break;
    case TimeTransformation::NONE:
        // Just copy data to the next buffer
        fn_compute_vect_->push_back(
            [=]()
            {
                cuComplex* buf = time_transformation_env_.gpu_p_acc_buffer.get();
                auto& q = time_transformation_env_.gpu_time_transformation_queue;
                size_t size = time_transformation_size * fd_.get_frame_res() * sizeof(cuComplex);

                cudaXMemcpyAsync(buf, q->get_data(), size, cudaMemcpyDeviceToDevice, stream_);
            });
        break;
    default:
        LOG_ERROR("Unknown time transformation");
        break;
    }
}

void FourierTransform::insert_stft()
{
    LOG_FUNC();

    fn_compute_vect_->push_back(
        [=]()
        {
            stft(time_transformation_env_.gpu_p_acc_buffer,
                 reinterpret_cast<cuComplex*>(time_transformation_env_.gpu_time_transformation_queue.get()->get_data()),
                 time_transformation_env_.stft_plan);
        });
}

void FourierTransform::insert_moments()
{
    LOG_FUNC();

    auto time_transformation_size = setting<settings::TimeTransformationSize>();
    size_t nyquist_index = time_transformation_size / 2;
    bool even = time_transformation_size % 2 == 0;
    fn_compute_vect_->push_back(
        [=]()
        {
            // compute the moment of order 0, corresponding to the sequence of frames multiplied by the
            // frequencies at order 0 (all equal to 1)
            tensor_multiply_vector_nyquist_compensation(moments_env_.moment0_buffer,
                                                        moments_env_.stft_res_buffer,
                                                        moments_env_.f0_buffer,
                                                        fd_.get_frame_res(),
                                                        moments_env_.f_start,
                                                        moments_env_.f_end,
                                                        nyquist_index,
                                                        even,
                                                        false,
                                                        stream_);

            // compute the moment of order 1, corresponding to the sequence of frames multiplied by the
            // frequencies at order 1
            tensor_multiply_vector_nyquist_compensation(moments_env_.moment1_buffer,
                                                        moments_env_.stft_res_buffer,
                                                        moments_env_.f1_buffer,
                                                        fd_.get_frame_res(),
                                                        moments_env_.f_start,
                                                        moments_env_.f_end,
                                                        nyquist_index,
                                                        even,
                                                        true,
                                                        stream_);

            // compute the moment of order 2, corresponding to the sequence of frames multiplied by the
            // frequencies at order 2
            tensor_multiply_vector_nyquist_compensation(moments_env_.moment2_buffer,
                                                        moments_env_.stft_res_buffer,
                                                        moments_env_.f2_buffer,
                                                        fd_.get_frame_res(),
                                                        moments_env_.f_start,
                                                        moments_env_.f_end,
                                                        nyquist_index,
                                                        even,
                                                        false,
                                                        stream_);
        });
}

void FourierTransform::insert_pca()
{
    LOG_FUNC();

    uint time_transformation_size = setting<settings::TimeTransformationSize>();
    cusolver_work_buffer_size_ = eigen_values_vectors_work_buffer_size(time_transformation_size);
    cusolver_work_buffer_.resize(cusolver_work_buffer_size_);

    fn_compute_vect_->push_back(
        [=]()
        {
            cuComplex* H = static_cast<cuComplex*>(time_transformation_env_.gpu_time_transformation_queue->get_data());
            cuComplex* cov = time_transformation_env_.pca_cov.get();
            cuComplex* V = nullptr;

            // cov = H' * H
            cov_matrix(H, static_cast<int>(fd_.get_frame_res()), time_transformation_size, cov);

            // Find eigen values and eigen vectors of cov
            // pca_eigen_values will contain sorted eigen values
            // cov and V will contain eigen vectors
            eigen_values_vectors(cov,
                                 time_transformation_size,
                                 time_transformation_env_.pca_eigen_values,
                                 &V,
                                 cusolver_work_buffer_,
                                 cusolver_work_buffer_size_,
                                 time_transformation_env_.pca_dev_info);

            // gpu_p_acc_buffer = H * V
            matrix_multiply_complex(H,
                                    V,
                                    static_cast<int>(fd_.get_frame_res()),
                                    time_transformation_size,
                                    time_transformation_size,
                                    time_transformation_env_.gpu_p_acc_buffer);
        });
}

void FourierTransform::insert_ssa_stft()
{
    LOG_FUNC();

    uint time_transformation_size = setting<settings::TimeTransformationSize>();

    cusolver_work_buffer_size_ = eigen_values_vectors_work_buffer_size(time_transformation_size);
    cusolver_work_buffer_.resize(cusolver_work_buffer_size_);

    static cuda_tools::CudaUniquePtr<cuComplex> tmp_matrix = nullptr;
    tmp_matrix.resize(time_transformation_size * time_transformation_size);

    fn_compute_vect_->push_back(
        [=]()
        {
            cuComplex* H = static_cast<cuComplex*>(time_transformation_env_.gpu_time_transformation_queue->get_data());
            cuComplex* cov = time_transformation_env_.pca_cov.get();
            cuComplex* V = nullptr;

            // cov = H' * H
            cov_matrix(H, static_cast<int>(fd_.get_frame_res()), time_transformation_size, cov);

            // pca_eigen_values = sorted eigen values of cov
            // cov and V = eigen vectors of cov
            eigen_values_vectors(cov,
                                 time_transformation_size,
                                 time_transformation_env_.pca_eigen_values,
                                 &V,
                                 cusolver_work_buffer_,
                                 cusolver_work_buffer_size_,
                                 time_transformation_env_.pca_dev_info);

            // filter eigen vectors
            // only keep vectors between q and q + q_acc
            ViewPQ q_struct = setting<settings::Q>();
            int q = q_struct.width != 0 ? q_struct.start : 0;
            int q_acc = q_struct.width != 0 ? q_struct.width : time_transformation_size;
            int q_index = q * time_transformation_size;
            int q_acc_index = q_acc * time_transformation_size;
            cudaXMemsetAsync(V, 0, q_index * sizeof(cuComplex), stream_);
            int copy_size = time_transformation_size * (time_transformation_size - (q + q_acc));
            cudaXMemsetAsync(V + q_index + q_acc_index, 0, copy_size * sizeof(cuComplex), stream_);

            // tmp = V * V'
            matrix_multiply_complex(V,
                                    V,
                                    time_transformation_size,
                                    time_transformation_size,
                                    time_transformation_size,
                                    tmp_matrix,
                                    CUBLAS_OP_N,
                                    CUBLAS_OP_C);

            // H = H * tmp
            matrix_multiply_complex(H,
                                    tmp_matrix,
                                    static_cast<int>(fd_.get_frame_res()),
                                    time_transformation_size,
                                    time_transformation_size,
                                    time_transformation_env_.gpu_p_acc_buffer);

            stft(time_transformation_env_.gpu_p_acc_buffer,
                 time_transformation_env_.gpu_p_acc_buffer,
                 time_transformation_env_.stft_plan);
        });
}

void FourierTransform::insert_store_p_frame()
{
    LOG_FUNC();

    fn_compute_vect_->push_back(
        [=]()
        {
            const int frame_res = static_cast<int>(fd_.get_frame_res());

            /* Copies with DeviceToDevice (which is the case here) are asynchronous
             * with respect to the host but never overlap with kernel execution*/
            cudaXMemcpyAsync(time_transformation_env_.gpu_p_frame,
                             (cuComplex*)time_transformation_env_.gpu_p_acc_buffer +
                                 setting<settings::P>().start * frame_res,
                             sizeof(cuComplex) * frame_res,
                             cudaMemcpyDeviceToDevice,
                             stream_);
        });
}

void FourierTransform::insert_time_transformation_cuts_view(const camera::FrameDescriptor& fd,
                                                            float* gpu_postprocess_frame_xz,
                                                            float* gpu_postprocess_frame_yz)
{
    LOG_FUNC();

    fn_compute_vect_->push_back(
        [=]()
        {
            if (setting<settings::CutsViewEnabled>())
            {
                ushort mouse_posx = 0;
                ushort mouse_posy = 0;

                // Conservation of the coordinates when cursor is outside of the
                // window
                auto x = setting<settings::X>();
                auto y = setting<settings::Y>();
                const ushort width = fd.width;
                const ushort height = fd.height;

                if (x.start < width && y.start < height)
                {
                    {
                        mouse_posx = x.start;
                        mouse_posy = y.start;
                    }

                    time_transformation_cuts_begin(gpu_postprocess_frame_xz,
                                                   gpu_postprocess_frame_yz,
                                                   time_transformation_env_.gpu_p_acc_buffer,
                                                   mouse_posx,
                                                   mouse_posy,
                                                   mouse_posx + x.width,
                                                   mouse_posy + y.width,
                                                   width,
                                                   height,
                                                   setting<settings::TimeTransformationSize>(),
                                                   setting<settings::XZ>().output_image_accumulation,
                                                   setting<settings::YZ>().output_image_accumulation,
                                                   setting<settings::ImageType>(),
                                                   stream_);
                }
            }
        });
}
