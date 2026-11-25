#include <sstream>
#include <stdexcept>

#include "fourier_transform.hh"

#include "cublas_handle.hh"
#include "cusolver_handle.hh"
#include "icompute.hh"

#include "tools_conversion.cuh"
#include "tools_compute.cuh"
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
#include "local_pca.hh"
#include "logger.hh"

using holovibes::FunctionVector;
using holovibes::Queue;
using holovibes::compute::FourierTransform;

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

    if (space_transformation == SpaceTransformation::FRESNELTR)
        insert_fresnel_transform();
    else
        insert_angular_spectrum(filter2d_enabled);

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
    case TimeTransformation::LOCAL_PCA:
        insert_local_pca();
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

    const auto frames = setting<settings::TimeTransformationSize>();
    const auto pixels = static_cast<int>(fd_.get_frame_res());
    const bool effective_rank = setting<settings::PcaEffectiveRankEnabled>();
    cusolver_work_buffer_size_ = eigen_values_vectors_work_buffer_size(frames);
    // Reuse the solver workspace for the temporal projector after diagonalization.
    const size_t projector_size = effective_rank ? static_cast<size_t>(frames) * frames : 0;
    if (!cusolver_work_buffer_.resize(std::max<size_t>(cusolver_work_buffer_size_, projector_size)))
        throw std::runtime_error("Insufficient GPU memory for PCA workspace");

    fn_compute_vect_->push_back(
        [=]()
        {
            const auto& queue = *time_transformation_env_.gpu_time_transformation_queue;
            cuComplex* H = static_cast<cuComplex*>(queue.get_data());
            cuComplex* output = time_transformation_env_.gpu_p_acc_buffer.get();
            const auto first_frame = effective_rank ? queue.get_start_index() : 0u;
            const auto copy_in_time_order = [frames, first_frame, stream = stream_](const cuComplex* source,
                                                                                  cuComplex* destination,
                                                                                  size_t elements_per_column)
            {
                const size_t prefix = first_frame * elements_per_column;
                const size_t suffix = (frames - first_frame) * elements_per_column;
                cudaXMemcpyAsync(destination, source + prefix, suffix * sizeof(cuComplex),
                                 cudaMemcpyDeviceToDevice, stream);
                if (prefix != 0)
                    cudaXMemcpyAsync(destination + suffix, source, prefix * sizeof(cuComplex),
                                     cudaMemcpyDeviceToDevice, stream);
            };

            const float factor = setting<settings::PcaEffectiveRankFactor>();
            if (effective_rank && factor == 0.f)
            {
                copy_in_time_order(H, output, pixels);
                return;
            }

            cuComplex* cov = time_transformation_env_.pca_cov.get();
            cuComplex* V = nullptr;

            if (effective_rank)
                pca_gram_matrix(cuda_tools::CublasHandle::instance(), H, pixels, frames, cov);
            else
                cov_matrix(H, pixels, frames, cov);

            eigen_values_vectors(cov,
                                 frames,
                                 time_transformation_env_.pca_eigen_values,
                                 &V,
                                 cusolver_work_buffer_,
                                 cusolver_work_buffer_size_,
                                 time_transformation_env_.pca_dev_info);

            const cuComplex* transform = V;
            if (effective_rank)
            {
                remove_pca_effective_rank(V, time_transformation_env_.pca_eigen_values, frames,
                                          time_transformation_env_.pca_dev_info, factor, stream_);
                // H V_keep V_keep^H restores the original temporal coordinates.
                cuComplex* projector = cusolver_work_buffer_.get();
                matrix_multiply_complex(V, V, frames, frames, frames, projector, CUBLAS_OP_N, CUBLAS_OP_C);
                transform = projector;
                if (first_frame != 0)
                {
                    // Reuse the covariance buffer to put the projector's columns in chronological order.
                    copy_in_time_order(projector, cov, frames);
                    transform = cov;
                }
            }
            matrix_multiply_complex(H, transform, pixels, frames, frames, output);
        });
}

void FourierTransform::insert_local_pca()
{
    // The callable owns this transform's scratch buffers. Refreshing the pipe
    // releases them, so the other time transforms never allocate Local PCA tiles.
    auto processor = std::make_shared<LocalPca>(fd_.width,
                                               fd_.height,
                                               setting<settings::TimeTransformationSize>(),
                                               setting<settings::PcaWidthX>(),
                                               setting<settings::PcaWidthY>(),
                                               setting<settings::PcaStrideX>(),
                                               setting<settings::PcaStrideY>(),
                                               stream_);
    fn_compute_vect_->push_back(
        [this, processor]()
        {
            auto& queue = time_transformation_env_.gpu_time_transformation_queue;
            processor->process(static_cast<const cuComplex*>(queue->get_data()),
                               time_transformation_env_.gpu_p_acc_buffer.get(),
                               setting<settings::PcaRankFactor>());
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
