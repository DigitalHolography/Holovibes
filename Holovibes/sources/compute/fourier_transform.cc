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
#include "shift_corners.cuh"
#include "apply_mask.cuh"
#include "svd.hh"

using holovibes::FunctionVector;
using holovibes::Queue;
using holovibes::compute::FourierTransform;

FourierTransform::FourierTransform(FunctionVector& fn_compute_vect,
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
    gpu_lens_.resize(fd_.get_frame_res());
}

void FourierTransform::insert_fft()
{
    if (cd_.filter2d_enabled)
    {
        update_filter2d_circles_mask(buffers_.gpu_filter2d_mask,
                                     fd_.width,
                                     fd_.height,
                                     cd_.filter2d_n1,
                                     cd_.filter2d_n2,
                                     cd_.filter2d_smooth_low,
                                     cd_.filter2d_smooth_high,
                                     stream_);

        // In FFT2 we do an optimisation to compute the filter2d in the same
        // reciprocal space to reduce the number of fft calculation
        if (cd_.space_transformation != SpaceTransformation::FFT2)
            insert_filter2d();
    }

    if (cd_.space_transformation == SpaceTransformation::FFT1)
        insert_fft1();
    else if (cd_.space_transformation == SpaceTransformation::FFT2)
        insert_fft2();
    if (cd_.space_transformation == SpaceTransformation::FFT1 || cd_.space_transformation == SpaceTransformation::FFT2)
        fn_compute_vect_.push_back([=]() { enqueue_lens(); });
}

void FourierTransform::insert_filter2d()
{

    fn_compute_vect_.push_back(
        [=]()
        {
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

    fft1_lens(gpu_lens_.get(), lens_side_size_, fd_.height, fd_.width, cd_.lambda, z, cd_.pixel_size, stream_);

    void* input_output = buffers_.gpu_spatial_transformation_buffer.get();

    fn_compute_vect_.push_back(
        [=]()
        {
            fft_1(static_cast<cuComplex*>(input_output),
                  static_cast<cuComplex*>(input_output),
                  cd_.batch_size,
                  gpu_lens_.get(),
                  spatial_transformation_plan_,
                  fd_.get_frame_res(),
                  stream_);
        });
}

void FourierTransform::insert_fft2()
{
    const float z = cd_.zdistance;

    fft2_lens(gpu_lens_.get(), lens_side_size_, fd_.height, fd_.width, cd_.lambda, z, cd_.pixel_size, stream_);

    shift_corners(gpu_lens_.get(), 1, fd_.width, fd_.height, stream_);

    if (cd_.filter2d_enabled)
        apply_mask(gpu_lens_.get(), buffers_.gpu_filter2d_mask.get(), fd_.width * fd_.height, 1, stream_);

    void* input_output = buffers_.gpu_spatial_transformation_buffer.get();

    fn_compute_vect_.push_back(
        [=]()
        {
            fft_2(static_cast<cuComplex*>(input_output),
                  static_cast<cuComplex*>(input_output),
                  cd_.batch_size,
                  gpu_lens_.get(),
                  spatial_transformation_plan_,
                  fd_,
                  stream_);
        });
}

std::unique_ptr<Queue>& FourierTransform::get_lens_queue()
{
    if (!gpu_lens_queue_ && cd_.lens_view_enabled)
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
        cuComplex* copied_lens_ptr = static_cast<cuComplex*>(gpu_lens_queue_->get_end());
        gpu_lens_queue_->enqueue(gpu_lens_, stream_);

        // For optimisation purposes, when FFT2 is activated, lens is shifted
        // We have to shift it again to ensure a good display
        if (cd_.space_transformation == SpaceTransformation::FFT2)
            shift_corners(copied_lens_ptr, 1, fd_.width, fd_.height, stream_);
        // Normalizing the newly enqueued element
        normalize_complex(copied_lens_ptr, fd_.get_frame_res(), stream_);
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
    else if (cd_.time_transformation == TimeTransformation::SSA_STFT)
    {
        insert_ssa_stft();
    }
    else // TimeTransformation::None
    {
        // Just copy data to the next buffer
        fn_compute_vect_.conditional_push_back(
            [=]()
            {
                cuComplex* buf = time_transformation_env_.gpu_p_acc_buffer.get();
                auto& q = time_transformation_env_.gpu_time_transformation_queue;
                size_t size = cd_.time_transformation_size * fd_.get_frame_res() * sizeof(cuComplex);

                cudaXMemcpyAsync(buf, q->get_data(), size, cudaMemcpyDeviceToDevice, stream_);
            });
    }
}

void FourierTransform::insert_stft()
{
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
    cusolver_work_buffer_size_ = eigen_values_vectors_work_buffer_size(cd_.time_transformation_size);
    cusolver_work_buffer_.resize(cusolver_work_buffer_size_);

    fn_compute_vect_.conditional_push_back(
        [=]()
        {
            cuComplex* H = static_cast<cuComplex*>(time_transformation_env_.gpu_time_transformation_queue->get_data());
            cuComplex* cov = time_transformation_env_.pca_cov.get();
            cuComplex* V = nullptr;

            // cov = H' * H
            cov_matrix(H, fd_.get_frame_res(), cd_.time_transformation_size, cov);

            // Find eigen values and eigen vectors of cov
            // pca_eigen_values will contain sorted eigen values
            // cov and V will contain eigen vectors
            eigen_values_vectors(cov,
                                 cd_.time_transformation_size,
                                 time_transformation_env_.pca_eigen_values,
                                 &V,
                                 cusolver_work_buffer_,
                                 cusolver_work_buffer_size_,
                                 time_transformation_env_.pca_dev_info);

            // gpu_p_acc_buffer = H * V
            matrix_multiply(H,
                            V,
                            fd_.get_frame_res(),
                            cd_.time_transformation_size,
                            cd_.time_transformation_size,
                            time_transformation_env_.gpu_p_acc_buffer);
        });
}

void FourierTransform::insert_ssa_stft()
{
    cusolver_work_buffer_size_ = eigen_values_vectors_work_buffer_size(cd_.time_transformation_size);
    cusolver_work_buffer_.resize(cusolver_work_buffer_size_);

    static cuda_tools::UniquePtr<cuComplex> tmp_matrix = nullptr;
    tmp_matrix.resize(cd_.time_transformation_size * cd_.time_transformation_size);

    fn_compute_vect_.conditional_push_back(
        [=]()
        {
            cuComplex* H = static_cast<cuComplex*>(time_transformation_env_.gpu_time_transformation_queue->get_data());
            cuComplex* cov = time_transformation_env_.pca_cov.get();
            cuComplex* V = nullptr;

            // cov = H' * H
            cov_matrix(H, fd_.get_frame_res(), cd_.time_transformation_size, cov);

            // pca_eigen_values = sorted eigen values of cov
            // cov and V = eigen vectors of cov
            eigen_values_vectors(cov,
                                 cd_.time_transformation_size,
                                 time_transformation_env_.pca_eigen_values,
                                 &V,
                                 cusolver_work_buffer_,
                                 cusolver_work_buffer_size_,
                                 time_transformation_env_.pca_dev_info);

            // filter eigen vectors
            // only keep vectors between q and q + q_acc
            int q = cd_.q.index.load();
            int q_acc = q != 0 ? cd_.q.accu_level.load() : cd_.time_transformation_size.load();
            int q_index = q * cd_.time_transformation_size;
            int q_acc_index = q_acc * cd_.time_transformation_size;
            cudaXMemsetAsync(V, 0, q_index * sizeof(cuComplex), stream_);
            int copy_size = cd_.time_transformation_size * (cd_.time_transformation_size - (q + q_acc));
            cudaXMemsetAsync(V + q_index + q_acc_index, 0, copy_size * sizeof(cuComplex), stream_);

            // tmp = V * V'
            matrix_multiply(V,
                            V,
                            cd_.time_transformation_size,
                            cd_.time_transformation_size,
                            cd_.time_transformation_size,
                            tmp_matrix,
                            CUBLAS_OP_N,
                            CUBLAS_OP_C);

            // H = H * tmp
            matrix_multiply(H,
                            tmp_matrix,
                            fd_.get_frame_res(),
                            cd_.time_transformation_size,
                            cd_.time_transformation_size,
                            time_transformation_env_.gpu_p_acc_buffer);

            stft(time_transformation_env_.gpu_p_acc_buffer,
                 time_transformation_env_.gpu_p_acc_buffer,
                 time_transformation_env_.stft_plan);
        });
}

void FourierTransform::insert_store_p_frame()
{
    fn_compute_vect_.conditional_push_back(
        [=]()
        {
            const int frame_res = fd_.get_frame_res();

            /* Copies with DeviceToDevice (which is the case here) are asynchronous
             * with respect to the host but never overlap with kernel execution*/
            cudaXMemcpyAsync(time_transformation_env_.gpu_p_frame,
                             (cuComplex*)time_transformation_env_.gpu_p_acc_buffer + cd_.p.index * frame_res,
                             sizeof(cuComplex) * frame_res,
                             cudaMemcpyDeviceToDevice,
                             stream_);
        });
}

void FourierTransform::insert_time_transformation_cuts_view()
{
    fn_compute_vect_.conditional_push_back(
        [=]()
        {
            if (cd_.time_transformation_cuts_enabled)
            {
                static ushort mouse_posx;
                static ushort mouse_posy;

                // Conservation of the coordinates when cursor is outside of the
                // window
                const ushort width = fd_.width;
                const ushort height = fd_.height;
                if (cd_.x.cuts < width && cd_.y.cuts < height)
                {
                    mouse_posx = cd_.x.cuts;
                    mouse_posy = cd_.y.cuts;
                }
                // -----------------------------------------------------
                time_transformation_cuts_begin(time_transformation_env_.gpu_p_acc_buffer,
                                               buffers_.gpu_postprocess_frame_xz.get(),
                                               buffers_.gpu_postprocess_frame_yz.get(),
                                               mouse_posx,
                                               mouse_posy,
                                               mouse_posx + cd_.x.accu_level.load(),
                                               mouse_posy + cd_.y.accu_level.load(),
                                               width,
                                               height,
                                               cd_.time_transformation_size,
                                               cd_.xz.img_accu_level.load(),
                                               cd_.yz.img_accu_level.load(),
                                               cd_.img_type.load(),
                                               stream_);
            }
        });
}
