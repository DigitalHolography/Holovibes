#include "registration.hh"
#include "apply_mask.cuh"
#include "convolution.cuh"
#include "shift_corners.cuh"
#include "tools_compute.cuh"
#include <cmath>
#include <opencv2/imgproc.hpp>

using holovibes::FunctionVector;
using holovibes::Queue;
using holovibes::compute::Registration;

struct Registration::LsciState
{
    cv::Mat frame;
    cv::Mat reference;
    cv::Mat mask;
    cv::Rect region;
    double reference_std = 0.;
};

void Registration::insert_registration(bool reset_reference)
{
    LOG_FUNC();

    if (reset_reference)
    {
        reference_ready_ = false;
        lsci_.reset();
        shift_x_ = shift_y_ = 0;
    }

    if (setting<settings::RegistrationEnabled>())
    {
        fn_compute_vect_->push_back(
            [=]()
            {
                if (setting<settings::TimeTransformation>() == TimeTransformation::LSCI)
                {
                    register_lsci();
                    return;
                }

                // Before computing the registration we need to have a reference image. Since `gpu_reference_image_` is
                // taken from the accumulation queue, we need to wait that the
                // `image_acc_env_.gpu_accumulation_xy_queue` is full and that we have a reference that is stable.
                if (!reference_ready_ || (image_acc_env_.gpu_accumulation_xy_queue.get() &&
                                          !image_acc_env_.gpu_accumulation_xy_queue.get()->is_full()))
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

void Registration::register_lsci()
{
    if (!lsci_)
        lsci_ = std::make_shared<LsciState>();
    auto& lsci = *lsci_;

    // Keep matching inexpensive on camera-sized maps. Only the matching copy is filtered.
    lsci.frame.create(fd_.height, fd_.width, CV_32F);
    cudaXMemcpyAsync(lsci.frame.ptr<float>(),
                     buffers_.gpu_postprocess_frame.get(),
                     fd_.width * fd_.height * sizeof(float),
                     cudaMemcpyDeviceToHost,
                     stream_);
    cudaXStreamSynchronize(stream_);

    const double scale = std::min(1., 768. / std::max(fd_.width, fd_.height));
    cv::Mat features;
    const cv::Size size(std::max(1, cvRound(fd_.width * scale)), std::max(1, cvRound(fd_.height * scale)));
    cv::resize(lsci.frame, features, size, 0, 0, cv::INTER_AREA);
    const int side = std::min(features.cols, features.rows);
    if (side < 8 || !cv::checkRange(features))
        return;

    // Suppress speckle and broad illumination changes so pulsating vessels drive the match.
    const double sigma = std::max(.5, 3. * side / 432.);
    cv::GaussianBlur(features, features, cv::Size(), sigma);
    cv::Mat background;
    cv::GaussianBlur(features, background, cv::Size(), std::max(2., 24. * side / 432.));
    features -= background;

    const int search_radius = std::max(1, cvRound(.14 * side));
    if (lsci.reference.empty())
    {
        const int radius = std::clamp(cvRound((side / 2 - search_radius - 1) * setting<settings::RegistrationZone>()),
                                      2,
                                      side / 2 - search_radius - 1);
        lsci.region = cv::Rect(features.cols / 2 - radius, features.rows / 2 - radius, 2 * radius + 1, 2 * radius + 1);
        lsci.mask.create(lsci.region.size(), CV_32F);
        for (int y = -radius; y <= radius; ++y)
            for (int x = -radius; x <= radius; ++x)
                lsci.mask.at<float>(y + radius, x + radius) =
                    std::clamp(static_cast<float>((1. - std::hypot(x, y) / radius) / .15), 0.f, 1.f);
    }

    cv::Scalar mean, deviation;
    cv::meanStdDev(features(lsci.region), mean, deviation, lsci.mask > 0);
    if (lsci.reference.empty())
    {
        // Wait for a usable first map rather than locking on a blank startup frame.
        if (deviation[0] > 1e-6)
        {
            lsci.reference = features(lsci.region).clone();
            lsci.reference_std = deviation[0];
        }
        return;
    }

    // During signal loss keep the last accepted shift, and recover against the same reference.
    if (deviation[0] >= .15 * lsci.reference_std)
    {
        const cv::Rect search(lsci.region.x - search_radius,
                              lsci.region.y - search_radius,
                              lsci.region.width + 2 * search_radius,
                              lsci.region.height + 2 * search_radius);
        cv::Mat correlation;
        cv::matchTemplate(features(search), lsci.reference, correlation, cv::TM_CCOEFF_NORMED, lsci.mask);
        for (int y = 0; y < correlation.rows; ++y)
            for (int x = 0; x < correlation.cols; ++x)
                if (!std::isfinite(correlation.at<float>(y, x)))
                    correlation.at<float>(y, x) = -1.f;

        double score;
        cv::Point peak;
        cv::minMaxLoc(correlation, nullptr, &score, nullptr, &peak);
        if (score >= .55)
        {
            // Parabolic peak refinement avoids quantizing movement to the reduced pixel size.
            auto refine = [](float left, float center, float right)
            {
                const float denominator = left - 2 * center + right;
                return std::abs(denominator) > 1e-10f ? std::clamp(.5f * (left - right) / denominator, -.5f, .5f) : 0.f;
            };
            const float dx = peak.x > 0 && peak.x + 1 < correlation.cols
                                 ? refine(correlation.at<float>(peak.y, peak.x - 1),
                                          correlation.at<float>(peak),
                                          correlation.at<float>(peak.y, peak.x + 1))
                                 : 0.f;
            const float dy = peak.y > 0 && peak.y + 1 < correlation.rows
                                 ? refine(correlation.at<float>(peak.y - 1, peak.x),
                                          correlation.at<float>(peak),
                                          correlation.at<float>(peak.y + 1, peak.x))
                                 : 0.f;
            shift_x_ = -cvRound((peak.x + dx - search_radius) * fd_.width / features.cols);
            shift_y_ = -cvRound((peak.y + dy - search_radius) * fd_.height / features.rows);
        }
    }
    shift_image(buffers_.gpu_postprocess_frame.get());
}

void Registration::image_preprocess(float* output, float* input, float* mean)
{
    get_mean_in_mask(input, gpu_circle_mask_, mean, fd_.width * fd_.height, stream_);
    rescale_in_mask(output, input, gpu_circle_mask_, *mean, fd_.width * fd_.height, stream_);
    apply_mask(output, gpu_circle_mask_, fd_.width * fd_.height, 1, stream_);
}

void Registration::set_gpu_reference_image()
{
    // LSCI captures its first usable, unscaled K map inside register_lsci().
    if (setting<settings::TimeTransformation>() == TimeTransformation::LSCI)
        return;

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
                reference_ready_ = true;
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
    lsci_.reset();
    shift_x_ = shift_y_ = 0;

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
