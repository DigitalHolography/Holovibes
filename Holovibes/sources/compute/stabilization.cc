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
                // Preprocessing the current image before the cross-correlation with the reference image.
                image_preprocess(gpu_current_image_, buffers_.gpu_postprocess_frame, &current_image_mean_);

                // Computing the cross-correlation of the reference image and the current image.
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

                // Getting the argmax of the xcorr2 output matrix.
                int max_index;
                int x;
                int y;
                matrix_argmax(gpu_xcorr_output_, fd_.width, fd_.height, max_index, x, y);

                // To keep the image in the display window.
                if (x > fd_.width / 2)
                    x -= fd_.width;
                if (y > fd_.height / 2)
                    y -= fd_.height;

                // Shifting the image to the computed (x,y) point.
                complex_translation(buffers_.gpu_postprocess_frame.get(), fd_.width, fd_.height, x, y, stream_);
            });
    }
}

void Stabilization::image_preprocess(float* output, float* input, float* mean)
{
    get_mean_in_mask(input, gpu_circle_mask_, mean, fd_.width * fd_.height, stream_);
    rescale_in_mask(output, input, gpu_circle_mask_, *mean, fd_.width * fd_.height, stream_);
    apply_mask(output, gpu_circle_mask_, fd_.width * fd_.height, 1, stream_);
}

void Stabilization::set_gpu_reference_image(float* new_gpu_reference_image_)
{
    if (setting<settings::StabilizationEnabled>())
    {
        cudaXMemcpyAsync(gpu_reference_image_,
                         new_gpu_reference_image_,
                         fd_.width * fd_.height * sizeof(float),
                         cudaMemcpyDeviceToDevice,
                         stream_);

        image_preprocess(gpu_reference_image_, gpu_reference_image_, &reference_image_mean_);
    }
}

void Stabilization::updade_cirular_mask()
{
    // Get the center and radius of the circle.
    float center_X = fd_.width / 2.0f;
    float center_Y = fd_.height / 2.0f;
    float radius =
        std::min(fd_.width, fd_.height) / 3.0f; // 3.0f could be change to get a different size for the circle.
    get_circular_mask(gpu_circle_mask_, center_X, center_Y, radius, fd_.width, fd_.height, stream_);
}