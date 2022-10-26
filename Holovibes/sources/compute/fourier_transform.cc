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

FourierTransform::FourierTransform(FunctionVector& fn_compute_vect,
                                   const holovibes::CoreBuffersEnv& buffers,
                                   const camera::FrameDescriptor& fd,
                                   holovibes::cuda_tools::CufftHandle& spatial_transformation_plan,
                                   holovibes::TimeTransformationEnv& time_transformation_env,
                                   const cudaStream_t& stream,
                                   AdvancedCache::Cache& advanced_cache,
                                   ComputeCache::Cache& compute_cache,
                                   ViewCache::Cache& view_cache)
    : gpu_lens_(nullptr)
    , lens_side_size_(std::max(fd.height, fd.width))
    , gpu_lens_queue_(nullptr)
    , fn_compute_vect_(fn_compute_vect)
    , buffers_(buffers)
    , fd_(fd)
    , spatial_transformation_plan_(spatial_transformation_plan)
    , time_transformation_env_(time_transformation_env)
    , stream_(stream)
    , advanced_cache_(advanced_cache)
    , compute_cache_(compute_cache)
    , view_cache_(view_cache)
{
    gpu_lens_.resize(fd_.get_frame_res());
}

void FourierTransform::insert_fft()
{
    LOG_FUNC();

    if (compute_cache_.get_value<Filter2D>().enabled)
    {
        update_filter2d_circles_mask(buffers_.gpu_filter2d_mask,
                                     fd_.width,
                                     fd_.height,
                                     compute_cache_.get_value<Filter2D>().n1,
                                     compute_cache_.get_value<Filter2D>().n2,
                                     advanced_cache_.get_value<Filter2DSmooth>().low,
                                     advanced_cache_.get_value<Filter2DSmooth>().high,
                                     stream_);

        // In FFT2 we do an optimisation to compute the filter2d in the same
        // reciprocal space to reduce the number of fft calculation
        if (compute_cache_.get_value<SpaceTransformation>() != SpaceTransformationEnum::FFT2)
            insert_filter2d();
    }

    if (compute_cache_.get_value<SpaceTransformation>() == SpaceTransformationEnum::FFT1)
        insert_fft1();
    else if (compute_cache_.get_value<SpaceTransformation>() == SpaceTransformationEnum::FFT2)
        insert_fft2();
    if (compute_cache_.get_value<SpaceTransformation>() == SpaceTransformationEnum::FFT1 ||
        compute_cache_.get_value<SpaceTransformation>() == SpaceTransformationEnum::FFT2)
        fn_compute_vect_.push_back([=]() { enqueue_lens(); });
}

void FourierTransform::insert_filter2d()
{
    LOG_FUNC();

    fn_compute_vect_.push_back(
        [=]()
        {
            filter2D(buffers_.gpu_spatial_transformation_buffer,
                     buffers_.gpu_filter2d_mask,
                     compute_cache_.get_value<BatchSize>(),
                     spatial_transformation_plan_,
                     fd_.width * fd_.height,
                     stream_);
        });
}

void FourierTransform::insert_fft1()
{
    LOG_FUNC(compute_cache_.get_value<Lambda>());

    const float z = compute_cache_.get_value<ZDistance>();

    fft1_lens(gpu_lens_.get(),
              lens_side_size_,
              fd_.height,
              fd_.width,
              compute_cache_.get_value<Lambda>(),
              z,
              compute_cache_.get_value<PixelSize>(),
              stream_);

    void* input_output = buffers_.gpu_spatial_transformation_buffer.get();

    fn_compute_vect_.push_back(
        [=]()
        {
            fft_1(static_cast<cuComplex*>(input_output),
                  static_cast<cuComplex*>(input_output),
                  compute_cache_.get_value<BatchSize>(),
                  gpu_lens_.get(),
                  spatial_transformation_plan_,
                  fd_.get_frame_res(),
                  stream_);
        });
}

void FourierTransform::insert_fft2()
{
    LOG_FUNC(compute_cache_.get_value<Lambda>());

    const float z = compute_cache_.get_value<ZDistance>();

    fft2_lens(gpu_lens_.get(),
              lens_side_size_,
              fd_.height,
              fd_.width,
              compute_cache_.get_value<Lambda>(),
              z,
              compute_cache_.get_value<PixelSize>(),
              stream_);

    shift_corners(gpu_lens_.get(), 1, fd_.width, fd_.height, stream_);

    if (compute_cache_.get_value<Filter2D>().enabled)
        apply_mask(gpu_lens_.get(), buffers_.gpu_filter2d_mask.get(), fd_.width * fd_.height, 1, stream_);

    void* input_output = buffers_.gpu_spatial_transformation_buffer.get();

    fn_compute_vect_.push_back(
        [=]()
        {
            fft_2(static_cast<cuComplex*>(input_output),
                  static_cast<cuComplex*>(input_output),
                  compute_cache_.get_value<BatchSize>(),
                  gpu_lens_.get(),
                  spatial_transformation_plan_,
                  fd_,
                  stream_);
        });
}

std::unique_ptr<Queue>& FourierTransform::get_lens_queue()
{
    LOG_FUNC();

    // FIXME WTF
    if (!gpu_lens_queue_)
    {
        auto fd = fd_;
        fd.depth = 8;
        gpu_lens_queue_ = std::make_unique<Queue>(fd, 16);
    }
    return gpu_lens_queue_;
}

// Inserted
void FourierTransform::enqueue_lens()
{
    // LOG-USELESS LOG_FUNC();

    if (gpu_lens_queue_)
    {
        // Getting the pointer in the location of the next enqueued element
        cuComplex* copied_lens_ptr = static_cast<cuComplex*>(gpu_lens_queue_->get_end());
        gpu_lens_queue_->enqueue(gpu_lens_, stream_);

        // For optimisation purposes, when FFT2 is activated, lens is shifted
        // We have to shift it again to ensure a good display
        if (compute_cache_.get_value<SpaceTransformation>() == SpaceTransformationEnum::FFT2)
            shift_corners(copied_lens_ptr, 1, fd_.width, fd_.height, stream_);
        // Normalizing the newly enqueued element
        normalize_complex(copied_lens_ptr, fd_.get_frame_res(), stream_);
    }
}

void FourierTransform::insert_time_transform()
{
    LOG_FUNC();

    if (compute_cache_.get_value<TimeTransformation>() == TimeTransformationEnum::STFT)
    {
        insert_stft();
    }
    else if (compute_cache_.get_value<TimeTransformation>() == TimeTransformationEnum::PCA)
    {
        insert_pca();
    }
    else if (compute_cache_.get_value<TimeTransformation>() == TimeTransformationEnum::SSA_STFT)
    {
        insert_ssa_stft();
    }
    else // TimeTransformationEnum::None
    {
        // Just copy data to the next buffer
        fn_compute_vect_.conditional_push_back(
            [=]()
            {
                cuComplex* buf = time_transformation_env_.gpu_p_acc_buffer.get();
                auto& q = time_transformation_env_.gpu_time_transformation_queue;
                size_t size =
                    compute_cache_.get_value<TimeTransformationSize>() * fd_.get_frame_res() * sizeof(cuComplex);

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

    uint time_transformation_size = compute_cache_.get_value<TimeTransformationSize>();
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

void FourierTransform::insert_ssa_stft()
{
    LOG_FUNC();

    uint time_transformation_size = compute_cache_.get_value<TimeTransformationSize>();

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
            ViewAccuPQ q_struct = view_cache_.get_value<ViewAccuQ>();
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
                                 view_cache_.get_value<ViewAccuP>().start * frame_res,
                             sizeof(cuComplex) * frame_res,
                             cudaMemcpyDeviceToDevice,
                             stream_);
        });
}

void FourierTransform::insert_time_transformation_cuts_view()
{
    LOG_FUNC();

    fn_compute_vect_.conditional_push_back(
        [=]()
        {
            if (view_cache_.get_value<CutsViewEnabled>())
            {
                ushort mouse_posx = 0;
                ushort mouse_posy = 0;

                // Conservation of the coordinates when cursor is outside of the
                // window
                const ushort width = fd_.width;
                const ushort height = fd_.height;

                ViewAccuXY x = view_cache_.get_value<ViewAccuX>();
                ViewAccuXY y = view_cache_.get_value<ViewAccuY>();
                if (x.start < width && y.start < height)
                {
                    {
                        mouse_posx = x.start;
                        mouse_posy = y.start;
                    }
                    // -----------------------------------------------------
                    time_transformation_cuts_begin(time_transformation_env_.gpu_p_acc_buffer,
                                                   buffers_.gpu_postprocess_frame_xz.get(),
                                                   buffers_.gpu_postprocess_frame_yz.get(),
                                                   mouse_posx,
                                                   mouse_posy,
                                                   mouse_posx + x.width,
                                                   mouse_posy + y.width,
                                                   width,
                                                   height,
                                                   compute_cache_.get_value<TimeTransformationSize>(),
                                                   view_cache_.get_value<ViewXZ>().output_image_accumulation,
                                                   view_cache_.get_value<ViewYZ>().output_image_accumulation,
                                                   view_cache_.get_value<ImageType>(),
                                                   stream_);
                }
            }
        });
}
