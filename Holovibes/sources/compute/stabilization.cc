#include "stabilization.hh"
#include "apply_mask.cuh"
#include "convolution.cuh"
#include "tools_compute.cuh"

using holovibes::FunctionVector;
using holovibes::Queue;
using holovibes::compute::Stabilization;

void Stabilization::insert_stabilization()
{
    LOG_FUNC();

    if (setting<settings::FftShiftEnabled>() && setting<settings::StabilizationEnabled>())
    {
        fn_compute_vect_.conditional_push_back(
            [=]()
            {
                get_mean_in_mask(buffers_.gpu_postprocess_frame,
                                 gpu_circle_mask_,
                                 &current_image_mean_,
                                 fd_.width * fd_.height,
                                 stream_);

                rescale_in_mask(gpu_current_image_,
                                buffers_.gpu_postprocess_frame,
                                gpu_circle_mask_,
                                current_image_mean_,
                                fd_.width * fd_.height,
                                stream_);

                apply_mask(gpu_current_image_, gpu_circle_mask_, fd_.width * fd_.height, 1, stream_);

                if (!ref)
                {
                    ref = true;

                    copy_(gpu_reference_image_, gpu_current_image_, fd_.width * fd_.height, stream_);

                    reference_image_mean_ = current_image_mean_;
                }
                else
                {
                    xcorr2(gpu_xcorr_output_,
                           gpu_current_image_,
                           gpu_reference_image_,
                           freq_size_,
                           stream_,
                           d_freq_1_,
                           d_freq_2_,
                           d_corr_freq_,
                           plan_2d_,
                           plan_2dinv_);

                    cudaStreamSynchronize(stream_);
                    int max_index;
                    cublasHandle_t handle;
                    cublasCreate(&handle);
                    cublasIsamax(handle, fd_.width * fd_.height, gpu_xcorr_output_, 1, &max_index);
                    max_index--;
                    cublasDestroy(handle);

                    // Step 4: Convert the linear index to (x, y) coordinates
                    int x = max_index % fd_.width; // Column
                    int y = max_index / fd_.width; // Row

                    if (x > fd_.width / 2)
                        x -= fd_.width;
                    if (y > fd_.height / 2)
                        y -= fd_.height;

                    complex_translation(buffers_.gpu_postprocess_frame.get(), fd_.width, fd_.height, x, y, stream_);
                }
            });
    }
}