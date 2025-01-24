#include "registration.hh"
#include "apply_mask.cuh"
#include "convolution.cuh"
#include "shift_corners.cuh"
#include "tools_compute.cuh"

using holovibes::FunctionVector;
using holovibes::Queue;
using holovibes::compute::Registration;

void Registration::insert_registration()
{
    LOG_FUNC();

    if (setting<settings::RegistrationEnabled>())
    {
        fn_compute_vect_->push_back(
            [=]()
            {
                // Before computing the registration we need to have a reference image. Since `gpu_reference_image_` is
                // taken from the accumulation queue, we need to wait that the
                // `image_acc_env_.gpu_accumulation_xy_queue` is full and that we have a reference that is stable.
                if (image_acc_env_.gpu_accumulation_xy_queue.get() &&
                    !image_acc_env_.gpu_accumulation_xy_queue.get()->is_full())
                    return;

                // Preprocessing the current image before the cross-correlation with the reference image.
                image_preprocess(gpu_current_image_, buffers_.gpu_postprocess_frame, &current_image_mean_);

                // Computing the cross-correlation of the reference image and the current image.
                xcorr2(gpu_xcorr_output_,
                       gpu_current_image_,
                       gpu_reference_image_,
                       d_freq_1_,
                       d_freq_2_,
                       plan_2d_,
                       plan_2dinv_,
                       freq_size_,
                       stream_);
                cudaXStreamSynchronize(stream_);

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

                // Shifting the image to the computed point. The shifted images is stored in `gpu_current_image_`.
                shift_x_ = -x;
                shift_y_ = -y;
                shift_image(buffers_.gpu_postprocess_frame);
            });
    }
}

void Registration::image_preprocess(float* output, float* input, float* mean)
{
    get_mean_in_mask(input, gpu_circle_mask_, mean, fd_.width * fd_.height, stream_);
    rescale_in_mask(output, input, gpu_circle_mask_, *mean, fd_.width * fd_.height, stream_);
    apply_mask(output, gpu_circle_mask_, fd_.width * fd_.height, 1, stream_);
}

void Registration::set_gpu_reference_image()
{
    if (setting<settings::RegistrationEnabled>())
    {
        ushort func_id = fn_compute_vect_->push_back(
            [=]
            {
                // The reference image is taken from the average of the image accumulation (done before this function).
                cudaXMemcpyAsync(gpu_reference_image_,
                                 buffers_.gpu_postprocess_frame,
                                 fd_.width * fd_.height * sizeof(float),
                                 cudaMemcpyDeviceToDevice,
                                 stream_);

                image_preprocess(gpu_reference_image_, gpu_reference_image_, &reference_image_mean_);
            });

        // After the `gpu_accumulation_xy_queue` buffer is full, we have a reference image on a fully accumulated image.
        // Then we can remove this function from the `fn_compute_vect_`, since we consider having a good enough
        // reference.
        fn_compute_vect_->conditionnal_remove(func_id,
                                              [this]
                                              {
                                                  return !image_acc_env_.gpu_accumulation_xy_queue.get() ||
                                                         image_acc_env_.gpu_accumulation_xy_queue.get()->is_full();
                                              });
    }
}

void Registration::updade_cirular_mask()
{
    // Get the center and radius of the circle.
    float center_X = fd_.width / 2.0f;
    float center_Y = fd_.height / 2.0f;
    float radius = (std::min(fd_.width, fd_.height) / 2) * setting<settings::RegistrationZone>();
    get_circular_mask(gpu_circle_mask_, center_X, center_Y, radius, fd_.width, fd_.height, stream_);

    // We shift the mask if the fftshift is not enabled to stay in the wanted zone (Useful for LDH).
    // Also, this happens only when we have a Fresnel Space Transform, since this process change the image shift.
    if ((setting<settings::SpaceTransformation>() == SpaceTransformation::FRESNELTR) !=
        setting<settings::FftShiftEnabled>())
        shift_corners(gpu_circle_mask_, 1, fd_.width, fd_.height, stream_);
}

void Registration::shift_image(float* input_output)
{
    circ_shift(gpu_current_image_, input_output, fd_.width, fd_.height, shift_x_, shift_y_, stream_);

    // Copy the result of the shift in `input_output` buffer.
    cudaXMemcpyAsync(input_output,
                     gpu_current_image_,
                     fd_.width * fd_.height * sizeof(float),
                     cudaMemcpyDeviceToDevice,
                     stream_);
}