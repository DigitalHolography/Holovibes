#include <sstream>

#include "fourier_transform.hh"

#include "cublas_handle.hh"
#include "cusolver_handle.hh"
#include "icompute.hh"

#include "tools_conversion.cuh"
#include "tools_compute.cuh"
#include "filter2D.cuh"
#include "fft1.cuh"
#include "fft2.cuh"
#include "transforms.cuh"
#include "stft.cuh"
#include "frame_reshape.cuh"
#include "cuda_tools/cufft_handle.hh"
#include "cuda_memory.cuh"
#include "queue.hh"
#include "shift_corners.cuh"
#include "apply_mask.cuh"
#include "svd.hh"
#include "logger.hh"

using holovibes::FunctionVector;
using holovibes::Queue;
using holovibes::compute::FourierTransform;

void FourierTransform::insert_fft(float* gpu_filter2d_mask,
                                  const uint width,
                                  const uint height,
                                  const uint radius_low,
                                  const uint radius_high,
                                  const uint smooth_low,
                                  const uint smooth_high,
                                  const SpaceTransformation space_transformation)
{
    LOG_FUNC();
    bool filter2d_enabled = setting<settings::Filter2dEnabled>();
    if (filter2d_enabled)
    {
        update_filter2d_circles_mask(gpu_filter2d_mask,
                                     width,
                                     height,
                                     radius_low,
                                     radius_high,
                                     smooth_low,
                                     smooth_high,
                                     stream_);

        // In FFT2 we do an optimisation to compute the filter2d in the same
        // reciprocal space to reduce the number of fft calculation
        if (space_transformation != SpaceTransformation::FFT2)
            insert_filter2d();
    }

    if (space_transformation == SpaceTransformation::FFT1)
        insert_fft1();
    else if (space_transformation == SpaceTransformation::FFT2)
        insert_fft2(filter2d_enabled);
    if (space_transformation == SpaceTransformation::FFT1 || space_transformation == SpaceTransformation::FFT2)
        fn_compute_vect_.push_back([=]() { enqueue_lens(space_transformation); });
}

void FourierTransform::insert_filter2d()
{
    LOG_FUNC();

    fn_compute_vect_.push_back(
        [=]()
        {
            filter2D(buffers_.gpu_spatial_transformation_buffer,
                     buffers_.gpu_filter2d_mask,
                     compute_cache_.get_batch_size(),
                     spatial_transformation_plan_,
                     fd_.width * fd_.height,
                     stream_);
        });
}

void FourierTransform::insert_fft1()
{
    LOG_FUNC();

    const float z = compute_cache_.get_z_distance();

    fft1_lens(gpu_lens_.get(),
              lens_side_size_,
              fd_.height,
              fd_.width,
              compute_cache_.get_lambda(),
              z,
              compute_cache_.get_pixel_size(),
              stream_);

    void* input_output = buffers_.gpu_spatial_transformation_buffer.get();

    fn_compute_vect_.push_back(
        [=]()
        {
            fft_1(static_cast<cuComplex*>(input_output),
                  static_cast<cuComplex*>(input_output),
                  compute_cache_.get_batch_size(),
                  gpu_lens_.get(),
                  spatial_transformation_plan_,
                  fd_.get_frame_res(),
                  stream_);
        });
}

void FourierTransform::insert_fft2(bool filter2d_enabled)
{
    LOG_FUNC();

    const float z = compute_cache_.get_z_distance();

    fft2_lens(gpu_lens_.get(),
              lens_side_size_,
              fd_.height,
              fd_.width,
              compute_cache_.get_lambda(),
              z,
              compute_cache_.get_pixel_size(),
              stream_);

    shift_corners(gpu_lens_.get(), 1, fd_.width, fd_.height, stream_);

    if (filter2d_enabled)
        apply_mask(gpu_lens_.get(), buffers_.gpu_filter2d_mask.get(), fd_.width * fd_.height, 1, stream_);

    void* input_output = buffers_.gpu_spatial_transformation_buffer.get();

    fn_compute_vect_.push_back(
        [=]()
        {
            fft_2(static_cast<cuComplex*>(input_output),
                  static_cast<cuComplex*>(input_output),
                  compute_cache_.get_batch_size(),
                  gpu_lens_.get(),
                  spatial_transformation_plan_,
                  fd_,
                  stream_);
        });
}

std::unique_ptr<Queue>& FourierTransform::get_lens_queue()
{
    LOG_FUNC();

    if (!gpu_lens_queue_)
    {
        auto fd = fd_;
        fd.depth = 8;
        gpu_lens_queue_ = std::make_unique<Queue>(fd, 16);
    }
    return gpu_lens_queue_;
}

// Inserted
void FourierTransform::enqueue_lens(SpaceTransformation space_transformation)
{
    // LOG-USELESS LOG_FUNC();

    if (gpu_lens_queue_)
    {
        // Getting the pointer in the location of the next enqueued element
        cuComplex* copied_lens_ptr = static_cast<cuComplex*>(gpu_lens_queue_->get_end());
        gpu_lens_queue_->enqueue(gpu_lens_, stream_);

        // For optimisation purposes, when FFT2 is activated, lens is shifted
        // We have to shift it again to ensure a good display
        if (space_transformation == SpaceTransformation::FFT2)
            shift_corners(copied_lens_ptr, 1, fd_.width, fd_.height, stream_);
        // Normalizing the newly enqueued element
        normalize_complex(copied_lens_ptr, fd_.get_frame_res(), stream_);
    }
}

void FourierTransform::insert_time_transform(const TimeTransformation time_transformation,
                                             const uint time_transformation_size)
{
    LOG_FUNC();

    if (time_transformation == TimeTransformation::STFT)
    {
        insert_stft();
    }
    else if (time_transformation == TimeTransformation::PCA)
    {
        insert_pca();
    }
    else if (time_transformation == TimeTransformation::SSA_STFT)
    {
        insert_ssa_stft(setting<settings::Q>());
    }
    else // TimeTransformation::None
    {
        // Just copy data to the next buffer
        fn_compute_vect_.conditional_push_back(
            [=]()
            {
                cuComplex* buf = time_transformation_env_.gpu_p_acc_buffer.get();
                auto& q = time_transformation_env_.gpu_time_transformation_queue;
                size_t size = time_transformation_size * fd_.get_frame_res() * sizeof(cuComplex);

                cudaXMemcpyAsync(buf, q->get_data(), size, cudaMemcpyDeviceToDevice, stream_);
            });
    }
}

void FourierTransform::insert_stft()
{
    LOG_FUNC();

    fn_compute_vect_.conditional_push_back(
        [=]()
        {
            stft(reinterpret_cast<cuComplex*>(time_transformation_env_.gpu_time_transformation_queue.get()->get_data()),
                 time_transformation_env_.gpu_p_acc_buffer,
                 time_transformation_env_.stft_plan);
        });
}

void FourierTransform::insert_pca()
{
    LOG_FUNC();

    uint time_transformation_size = compute_cache_.get_time_transformation_size();
    cusolver_work_buffer_size_ = eigen_values_vectors_work_buffer_size(time_transformation_size);
    cusolver_work_buffer_.resize(cusolver_work_buffer_size_);

    fn_compute_vect_.conditional_push_back(
        [=]()
        {
            cuComplex* H = static_cast<cuComplex*>(time_transformation_env_.gpu_time_transformation_queue->get_data());
            cuComplex* cov = time_transformation_env_.pca_cov.get();
            cuComplex* V = nullptr;

            // cov = H' * H
            cov_matrix(H, fd_.get_frame_res(), time_transformation_size, cov);

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
            matrix_multiply(H,
                            V,
                            fd_.get_frame_res(),
                            time_transformation_size,
                            time_transformation_size,
                            time_transformation_env_.gpu_p_acc_buffer);
        });
}

void FourierTransform::insert_ssa_stft(ViewPQ view_q)
{
    LOG_FUNC();

    uint time_transformation_size = compute_cache_.get_time_transformation_size();

    cusolver_work_buffer_size_ = eigen_values_vectors_work_buffer_size(time_transformation_size);
    cusolver_work_buffer_.resize(cusolver_work_buffer_size_);

    static cuda_tools::UniquePtr<cuComplex> tmp_matrix = nullptr;
    tmp_matrix.resize(time_transformation_size * time_transformation_size);

    fn_compute_vect_.conditional_push_back(
        [=]()
        {
            cuComplex* H = static_cast<cuComplex*>(time_transformation_env_.gpu_time_transformation_queue->get_data());
            cuComplex* cov = time_transformation_env_.pca_cov.get();
            cuComplex* V = nullptr;

            // cov = H' * H
            cov_matrix(H, fd_.get_frame_res(), time_transformation_size, cov);

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
            ViewPQ q_struct = view_q;
            int q = q_struct.width != 0 ? q_struct.start : 0;
            int q_acc = q_struct.width != 0 ? q_struct.width : time_transformation_size;
            int q_index = q * time_transformation_size;
            int q_acc_index = q_acc * time_transformation_size;
            cudaXMemsetAsync(V, 0, q_index * sizeof(cuComplex), stream_);
            int copy_size = time_transformation_size * (time_transformation_size - (q + q_acc));
            cudaXMemsetAsync(V + q_index + q_acc_index, 0, copy_size * sizeof(cuComplex), stream_);

            // tmp = V * V'
            matrix_multiply(V,
                            V,
                            time_transformation_size,
                            time_transformation_size,
                            time_transformation_size,
                            tmp_matrix,
                            CUBLAS_OP_N,
                            CUBLAS_OP_C);

            // H = H * tmp
            matrix_multiply(H,
                            tmp_matrix,
                            fd_.get_frame_res(),
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

    fn_compute_vect_.conditional_push_back(
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
                                                            float* gpu_postprocess_frame_yz,
                                                            uint time_transformation_size)
{
    LOG_FUNC();

    fn_compute_vect_.conditional_push_back(
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
                    // -----------------------------------------------------
                    time_transformation_cuts_begin(time_transformation_env_.gpu_p_acc_buffer,
                                                   gpu_postprocess_frame_xz,
                                                   gpu_postprocess_frame_yz,
                                                   mouse_posx,
                                                   mouse_posy,
                                                   mouse_posx + x.width,
                                                   mouse_posy + y.width,
                                                   width,
                                                   height,
                                                   time_transformation_size,
                                                   setting<settings::XZ>().output_image_accumulation,
                                                   setting<settings::YZ>().output_image_accumulation,
                                                   setting<settings::ImageType>(),
                                                   stream_);
                }
            }
        });
}
