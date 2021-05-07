/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#include <sstream>

#include "fourier_transform.hh"
#include "compute_descriptor.hh"
#include "cublas_handle.hh"
#include "cusolver_handle.hh"
#include "icompute.hh"

#include "tools_conversion.cuh"
#include "tools_compute.cuh"
#include "filter2d.cuh"
#include "fft1.cuh"
#include "fft2.cuh"
#include "transforms.cuh"
#include "stft.cuh"
#include "frame_reshape.cuh"
#include "cuda_tools/cufft_handle.hh"
#include "cuda_memory.cuh"
#include "queue.hh"

using holovibes::FunctionVector;
using holovibes::Queue;
using holovibes::compute::FourierTransform;

FourierTransform::FourierTransform(
    FunctionVector& fn_compute_vect,
    const holovibes::CoreBuffersEnv& buffers,
    const camera::FrameDescriptor& fd,
    holovibes::ComputeDescriptor& cd,
    holovibes::cuda_tools::CufftHandle& spatial_transformation_plan,
    const holovibes::BatchEnv& batch_env,
    holovibes::TimeTransformationEnv& time_transformation_env,
    const cudaStream_t& stream)
    : gpu_lens_(nullptr)
    , lens_side_size_(std::max(fd.height, fd.width))
    , gpu_lens_queue_(nullptr)
    , fn_compute_vect_(fn_compute_vect)
    , buffers_(buffers)
    , fd_(fd)
    , cd_(cd)
    , spatial_transformation_plan_(spatial_transformation_plan)
    , batch_env_(batch_env)
    , time_transformation_env_(time_transformation_env)
    , stream_(stream)
{
    gpu_lens_.resize(fd_.frame_res());
}

void FourierTransform::insert_fft()
{
    if (cd_.space_transformation != SpaceTransformation::FFT2 &&
        cd_.filter2d_enabled)
        insert_filter2d();

    if (cd_.space_transformation == SpaceTransformation::FFT1)
        insert_fft1();
    else if (cd_.space_transformation == SpaceTransformation::FFT2)
        insert_fft2();
    if (cd_.space_transformation == SpaceTransformation::FFT1 ||
        cd_.space_transformation == SpaceTransformation::FFT2)
        fn_compute_vect_.push_back([=]() { enqueue_lens(); });
}

void FourierTransform::insert_filter2d()
{
    fn_compute_vect_.push_back([=]() {
        filter2D(buffers_.gpu_spatial_transformation_buffer,
                 buffers_.gpu_filter2d_mask,
                 cd_.batch_size,
                 spatial_transformation_plan_,
                 fd_.width * fd_.height,
                 stream_);
    });
}

void FourierTransform::insert_fft1()
{
    const float z = cd_.zdistance;

    fft1_lens(gpu_lens_.get(),
              lens_side_size_,
              fd_.height,
              fd_.width,
              cd_.lambda,
              z,
              cd_.pixel_size,
              stream_);

    fn_compute_vect_.push_back([=]() {
        fft_1(buffers_.gpu_spatial_transformation_buffer,
              buffers_.gpu_spatial_transformation_buffer,
              cd_.batch_size,
              gpu_lens_.get(),
              spatial_transformation_plan_,
              fd_.frame_res(),
              stream_);
    });
}

void FourierTransform::insert_fft2()
{
    const float z = cd_.zdistance;

    fft2_lens(gpu_lens_.get(),
              lens_side_size_,
              fd_.height,
              fd_.width,
              cd_.lambda,
              z,
              cd_.pixel_size,
              stream_);

    fn_compute_vect_.push_back([=]() {
        fft_2(buffers_.gpu_spatial_transformation_buffer,
              buffers_.gpu_spatial_transformation_buffer,
              cd_.batch_size,
              buffers_.gpu_filter2d_mask,
              cd_.filter2d_enabled,
              gpu_lens_.get(),
              spatial_transformation_plan_,
              fd_,
              stream_);
    });
}

std::unique_ptr<Queue>& FourierTransform::get_lens_queue()
{
    if (!gpu_lens_queue_ && cd_.gpu_lens_display_enabled)
    {
        auto fd = fd_;
        fd.depth = 8;
        gpu_lens_queue_ = std::make_unique<Queue>(fd, 16);
    }
    return gpu_lens_queue_;
}

void FourierTransform::enqueue_lens()
{
    if (gpu_lens_queue_)
    {
        // Getting the pointer in the location of the next enqueued element
        cuComplex* copied_lens_ptr =
            static_cast<cuComplex*>(gpu_lens_queue_->get_end());
        gpu_lens_queue_->enqueue(gpu_lens_, stream_);
        // Normalizing the newly enqueued element
        normalize_complex(copied_lens_ptr, fd_.frame_res(), stream_);
    }
}

void FourierTransform::insert_time_transform()
{
    if (cd_.time_transformation == TimeTransformation::STFT)
    {
        insert_stft();
    }
    else if (cd_.time_transformation == TimeTransformation::PCA)
    {
        insert_pca();
    }
    else // TimeTransformation::None
    {
        // Just copy data to the next buffer
        fn_compute_vect_.conditional_push_back([=]() {
            cuComplex* buf = time_transformation_env_.gpu_p_acc_buffer.get();
            auto& q = time_transformation_env_.gpu_time_transformation_queue;
            size_t size = cd_.time_transformation_size * fd_.frame_res() *
                          sizeof(cuComplex);

            cudaXMemcpyAsync(buf,
                             q->get_data(),
                             size,
                             cudaMemcpyDeviceToDevice,
                             stream_);
        });
    }
}

void FourierTransform::insert_stft()
{
    fn_compute_vect_.conditional_push_back([=]() {
        stft(time_transformation_env_.gpu_time_transformation_queue.get(),
             time_transformation_env_.gpu_p_acc_buffer,
             time_transformation_env_.stft_plan);
    });
}

void FourierTransform::insert_pca()
{
    constexpr uint sample_step = 16;
    cusolver_work_buffer_size_ = 0;
    cusolverSafeCall(
        cusolverDnCheevd_bufferSize(cuda_tools::CusolverHandle::instance(),
                                    CUSOLVER_EIG_MODE_VECTOR,
                                    CUBLAS_FILL_MODE_LOWER,
                                    cd_.time_transformation_size,
                                    nullptr,
                                    cd_.time_transformation_size,
                                    nullptr,
                                    &cusolver_work_buffer_size_));
    cusolver_work_buffer_.resize(cusolver_work_buffer_size_);

    subsample_pca_buffer_.resize(cd_.time_transformation_size *
                                 fd_.frame_res() / (sample_step * sample_step));

    fn_compute_vect_.conditional_push_back([=]() {
        unsigned short p_acc = cd_.p_accu_enabled ? cd_.p_acc_level + 1 : 1;
        unsigned short p = cd_.pindex;
        if (p + p_acc > cd_.time_transformation_size)
        {
            p_acc = cd_.time_transformation_size - p;
        }

        constexpr cuComplex alpha{1, 0};
        constexpr cuComplex beta{0, 0};

        cuComplex* image_data = static_cast<cuComplex*>(
            time_transformation_env_.gpu_time_transformation_queue->get_data());

        // cuComplex* subsample_data = subsample_pca_buffer_.get();
        // subsample_frame_complex_batched(image_data,
        //                                 fd_.width,
        //                                 fd_.height,
        //                                 subsample_data,
        //                                 sample_step,
        //                                 cd_.time_transformation_size,
        //                                 stream_);

        cuComplex* H = image_data;
        uint frame_res = fd_.frame_res();
        // cuComplex* H = subsample_data;
        // uint frame_res = fd_.frame_res() / (sample_step * sample_step);
        cuComplex* cov = time_transformation_env_.pca_cov.get();

        // cov = H' * H
        cublasSafeCall(cublasCgemm3m(cuda_tools::CublasHandle::instance(),
                                     CUBLAS_OP_C,
                                     CUBLAS_OP_N,
                                     cd_.time_transformation_size,
                                     cd_.time_transformation_size,
                                     frame_res,
                                     &alpha,
                                     H,
                                     frame_res,
                                     H,
                                     frame_res,
                                     &beta,
                                     cov,
                                     cd_.time_transformation_size));

        // Find eigen values and eigen vectors of cov
        // pca_eigen_values will contain sorted eigen values
        // cov will contain eigen vectors
        cusolverSafeCall(
            cusolverDnCheevd(cuda_tools::CusolverHandle::instance(),
                             CUSOLVER_EIG_MODE_VECTOR,
                             CUBLAS_FILL_MODE_LOWER,
                             cd_.time_transformation_size,
                             cov,
                             cd_.time_transformation_size,
                             time_transformation_env_.pca_eigen_values.get(),
                             cusolver_work_buffer_.get(),
                             cusolver_work_buffer_size_,
                             time_transformation_env_.pca_dev_info.get()));

        // eigen vectors
        cuComplex* V = cov;

        // H = H * V
        cublasSafeCall(
            cublasCgemm3m(cuda_tools::CublasHandle::instance(),
                          CUBLAS_OP_N,
                          CUBLAS_OP_N,
                          fd_.frame_res(),
                          cd_.time_transformation_size,
                          cd_.time_transformation_size,
                          &alpha,
                          image_data,
                          fd_.frame_res(),
                          V,
                          cd_.time_transformation_size,
                          &beta,
                          time_transformation_env_.gpu_p_acc_buffer.get(),
                          fd_.frame_res()));
    });
}

void FourierTransform::insert_store_p_frame()
{
    fn_compute_vect_.conditional_push_back([=]() {
        const int frame_res = fd_.frame_res();

        /* Copies with DeviceToDevice (which is the case here) are asynchronous
         * with respect to the host but never overlap with kernel execution*/
        cudaXMemcpyAsync(time_transformation_env_.gpu_p_frame,
                         (cuComplex*)time_transformation_env_.gpu_p_acc_buffer +
                             cd_.pindex * frame_res,
                         sizeof(cuComplex) * frame_res,
                         cudaMemcpyDeviceToDevice,
                         stream_);
    });
}

void FourierTransform::insert_time_transformation_cuts_view()
{
    fn_compute_vect_.conditional_push_back([=]() {
        if (cd_.time_transformation_cuts_enabled)
        {
            static ushort mouse_posx;
            static ushort mouse_posy;

            // Conservation of the coordinates when cursor is outside of the
            // window
            units::PointFd cursorPos = cd_.getStftCursor();
            const ushort width = fd_.width;
            const ushort height = fd_.height;
            if (static_cast<ushort>(cursorPos.x()) < width &&
                static_cast<ushort>(cursorPos.y()) < height)
            {
                mouse_posx = cursorPos.x();
                mouse_posy = cursorPos.y();
            }
            // -----------------------------------------------------
            time_transformation_cuts_begin(
                time_transformation_env_.gpu_p_acc_buffer,
                buffers_.gpu_postprocess_frame_xz.get(),
                buffers_.gpu_postprocess_frame_yz.get(),
                mouse_posx,
                mouse_posy,
                mouse_posx + (cd_.x_accu_enabled ? cd_.x_acc_level.load() : 0),
                mouse_posy + (cd_.y_accu_enabled ? cd_.y_acc_level.load() : 0),
                width,
                height,
                cd_.time_transformation_size,
                cd_.img_acc_slice_xz_enabled ? cd_.img_acc_slice_xz_level.load()
                                             : 1,
                cd_.img_acc_slice_yz_enabled ? cd_.img_acc_slice_yz_level.load()
                                             : 1,
                cd_.img_type.load(),
                stream_);
        }
    });
}
