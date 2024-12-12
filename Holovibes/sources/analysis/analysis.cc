#include "convolution.cuh"
#include "cuda_memory.cuh"
#include "shift_corners.cuh"
#include "holovibes.hh"
#include "matrix_operations.hh"
#include "moments_treatments.cuh"
#include "vesselness_filter.cuh"
#include "barycentre.cuh"
#include "vascular_pulse.cuh"
#include "tools_analysis.cuh"
#include "API.hh"
#include "otsu.cuh"
#include "cublas_handle.hh"
#include "bw_area.cuh"
#include "circular_video_buffer.hh"
#include "segment_vessels.cuh"
#include "tools_analysis_debug.hh"
#include "tools_compute.cuh"
#include "map.cuh"
#include "chart_mean_vessels.cuh"

#define OTSU_BINS 256

namespace holovibes::analysis
{

#pragma region Getters

float* Analysis::get_mask_result() { return mask_result_buffer_.get(); }

size_t Analysis::get_mask_nnz() { return count_non_zero(mask_result_buffer_, fd_.width, fd_.height, stream_); }

#pragma endregion

#pragma region Compute

// To be deleted
void Analysis::insert_bin_moments()
{
    cudaXMemcpyAsync(moments_env_.moment0_buffer,
                     m0_bin_video_ + i_ * 512 * 512,
                     sizeof(float) * 512 * 512,
                     cudaMemcpyDeviceToDevice,
                     stream_);
    cudaXMemcpyAsync(moments_env_.moment1_buffer,
                     m1_bin_video_ + i_ * 512 * 512,
                     sizeof(float) * 512 * 512,
                     cudaMemcpyDeviceToDevice,
                     stream_);
    i_ = (i_ + 1) % 506;

    cudaXMemcpyAsync(buffers_.gpu_postprocess_frame,
                     moments_env_.moment0_buffer,
                     sizeof(float) * fd_.width * fd_.height,
                     cudaMemcpyDeviceToDevice,
                     stream_);
}

void Analysis::compute_pretreatment()
{
    // Compute the flat field corrected image for each frame of the video
    normalize_array(buffers_.gpu_postprocess_frame, fd_.get_frame_res(), 0, 255, stream_);

    // Fill the m0_ff_video circular buffer with the flat field corrected images and compute the mean from
    // it
    vesselness_mask_env_.m0_ff_video_cb_->add_new_frame(buffers_.gpu_postprocess_frame);
    vesselness_mask_env_.m0_ff_video_cb_->compute_mean_image();

    // Compute the centered image from the temporal mean of the video
    image_centering(vesselness_mask_env_.m0_ff_video_centered_,
                    vesselness_mask_env_.m0_ff_video_cb_->get_data_ptr(),
                    vesselness_mask_env_.m0_ff_video_cb_->get_mean_image(),
                    buffers_.gpu_postprocess_frame_size,
                    vesselness_mask_env_.m0_ff_video_cb_->get_frame_count(),
                    stream_);
}

void Analysis::compute_vesselness_response()
{
    // Compute the first vesselness mask which represents both vessels types (arteries and veins)
    vesselness_filter(buffers_.gpu_postprocess_frame,
                      vesselness_mask_env_.m0_ff_video_cb_->get_mean_image(),
                      setting<settings::VesselnessSigma>(),
                      vesselness_mask_env_.g_xx_mul_,
                      vesselness_mask_env_.g_xy_mul_,
                      vesselness_mask_env_.g_yy_mul_,
                      vesselness_mask_env_.kernel_x_size_,
                      vesselness_mask_env_.kernel_y_size_,
                      buffers_.gpu_postprocess_frame_size,
                      vesselness_filter_struct_,
                      cublas_handler_,
                      stream_);

    // Uncomment if using real moments
    shift_corners(buffers_.gpu_postprocess_frame.get(), 1, fd_.width, fd_.height, stream_);

    // Compute and apply a circular diaphragm mask on the vesselness output
    apply_diaphragm_mask(buffers_.gpu_postprocess_frame,
                         fd_.width / 2 - 1,
                         fd_.height / 2 - 1,
                         setting<settings::DiaphragmFactor>() * (fd_.width + fd_.height) / 2,
                         fd_.width,
                         fd_.height,
                         stream_);

    // From here ~160 FPS
    // Use otsu to get a binarise image, which gives us the vesselness mask
    compute_binarise_otsu(buffers_.gpu_postprocess_frame,
                          otsu_env_.otsu_histo_buffer_.get(),
                          fd_.width,
                          fd_.height,
                          stream_);

    // Store mask_vesselness for later computations
    cudaXMemcpyAsync(vesselness_mask_env_.mask_vesselness_,
                     buffers_.gpu_postprocess_frame,
                     sizeof(float) * buffers_.gpu_postprocess_frame_size,
                     cudaMemcpyDeviceToDevice,
                     stream_);
}

void Analysis::compute_barycentres_and_circle_mask()
{
    // Compute f_AVG_mean, which is the temporal average of M1 / M0
    // First, M1 is stored in a buffer so it is not overwritten
    cudaXMemcpyAsync(vesselness_mask_env_.m1_divided_by_m0_frame_,
                     moments_env_.moment1_buffer,
                     buffers_.gpu_postprocess_frame_size * sizeof(float),
                     cudaMemcpyDeviceToDevice,
                     stream_);
    // Then it is divided in place by M0
    divide_frames_inplace(vesselness_mask_env_.m1_divided_by_m0_frame_,
                          moments_env_.moment0_buffer,
                          buffers_.gpu_postprocess_frame_size,
                          stream_);
    // A circular buffer is used to store f_AVG_mean, which is M1 / M0
    vesselness_mask_env_.f_avg_video_cb_->add_new_frame(vesselness_mask_env_.m1_divided_by_m0_frame_);

    vesselness_mask_env_.f_avg_video_cb_->compute_mean_image();

    // From here ~111 FPS
    // Compute vascular image
    compute_hadamard_product(vesselness_mask_env_.vascular_image_,
                             vesselness_mask_env_.m0_ff_video_cb_->get_mean_image(),
                             vesselness_mask_env_.f_avg_video_cb_->get_mean_image(),
                             buffers_.gpu_postprocess_frame_size,
                             stream_);

    // From here ~110 FPS
    // Apply gaussian blur
    apply_convolution(vesselness_mask_env_.vascular_image_,
                      vesselness_mask_env_.vascular_kernel_,
                      fd_.width,
                      fd_.height,
                      vesselness_mask_env_.vascular_kernel_size_,
                      vesselness_mask_env_.vascular_kernel_size_,
                      vesselness_filter_struct_.convolution_tmp_buffer,
                      stream_,
                      ConvolutionPaddingType::SCALAR,
                      0);

    // From here ~105 FPS
    // Compute the barycentre and the CRV
    compute_barycentre_circle_mask(vesselness_mask_env_.circle_mask_,
                                   vesselness_filter_struct_.CRV_circle_mask,
                                   vesselness_mask_env_.vascular_image_,
                                   fd_.width,
                                   fd_.height,
                                   setting<settings::BarycenterFactor>(),
                                   stream_);

    // From here ~90 FPS
    // The result of otsu is now used here to get the mask vesselness clean
    // The formula is : maskVesselness & bwareafilt(maskVesselness | cercleMask);
    cudaXMemcpyAsync(vesselness_mask_env_.bwareafilt_result_,
                     buffers_.gpu_postprocess_frame, // This is the result of
                                                     // otsu
                     sizeof(float) * buffers_.gpu_postprocess_frame_size,
                     cudaMemcpyDeviceToDevice,
                     stream_);
    apply_mask_or(vesselness_mask_env_.bwareafilt_result_,
                  vesselness_mask_env_.circle_mask_,
                  fd_.width,
                  fd_.height,
                  stream_);
    bwareafilt(vesselness_mask_env_.bwareafilt_result_,
               fd_.width,
               fd_.height,
               bw_area_env_.uint_buffer_1_,
               bw_area_env_.uint_buffer_2_,
               bw_area_env_.float_buffer_,
               bw_area_env_.size_t_gpu_,
               cublas_handler_,
               stream_);
    cudaXMemcpyAsync(vesselness_mask_env_.mask_vesselness_clean_,
                     buffers_.gpu_postprocess_frame, // this is the result of
                                                     // otsu
                     sizeof(float) * buffers_.gpu_postprocess_frame_size,
                     cudaMemcpyDeviceToDevice,
                     stream_);
    apply_mask_and(vesselness_mask_env_.mask_vesselness_clean_,
                   vesselness_mask_env_.bwareafilt_result_,
                   fd_.width,
                   fd_.height,
                   stream_);
}

void Analysis::compute_correlation()
{
    // Fps here ~90 FPS
    // The mean [1 2] is compute here using the multiplication of M0_ff_video and mask_vesselness_clean
    // vascularPulse = mean(M0_ff_video .* maskVesselnessClean, [1 2]);
    vesselness_mask_env_.m0_ff_video_cb_->compute_mean_1_2(vesselness_mask_env_.mask_vesselness_clean_);
    // Fps here ~45 FPS, better than before (5 FPS) (this is here we loose perf)
    cudaXMemcpy(vesselness_filter_struct_.vascular_pulse,
                vesselness_mask_env_.m0_ff_video_cb_->get_mean_1_2_(),
                vesselness_mask_env_.m0_ff_video_cb_->get_frame_count() * sizeof(float),
                cudaMemcpyDeviceToDevice);

    // To get the new vascular_pulse, we need to do that : vascularPulse = vascularPulse ./ nnz(maskVesselnessClean), so
    // we compute nnz
    int nnz = count_non_zero(vesselness_mask_env_.mask_vesselness_clean_, fd_.height, fd_.width, stream_);
    // then this function will directly returns us the R_vascular_pulse, which is
    // vascularPulse_centered = vascularPulse - mean(vascularPulse, 3);
    // R_VascularPulse = mean(M0_ff_video_centered .* vascularPulse_centered, 3) ./ (std((M0_ff_video_centered), [], 3)
    // * std(vascularPulse_centered, [], 3));
    compute_first_correlation(vesselness_mask_env_.R_vascular_pulse_,
                              vesselness_mask_env_.m0_ff_video_centered_,
                              vesselness_filter_struct_.vascular_pulse,
                              nnz,
                              vesselness_mask_env_.m0_ff_video_cb_->get_frame_count(),
                              vesselness_filter_struct_,
                              buffers_.gpu_postprocess_frame_size,
                              stream_);
}

void Analysis::compute_segment_vessels()
{
    // compute the thresholds using otsu method, which will give us an image with 5 values : 1, 2, 3, 4 or 5
    // to compute the thresholds, we will need the multiplication of R_vascular_pulse and mask_vesselness_clean
    cudaXMemcpyAsync(vesselness_mask_env_.before_threshold,
                     vesselness_mask_env_.R_vascular_pulse_,
                     sizeof(float) * buffers_.gpu_postprocess_frame_size,
                     cudaMemcpyDeviceToDevice,
                     stream_);
    apply_mask_and(vesselness_mask_env_.before_threshold,
                   vesselness_mask_env_.mask_vesselness_clean_,
                   fd_.width,
                   fd_.height,
                   stream_);

    float thresholds[3] = {0.207108953480839f, 0.334478400506137f, 0.458741275652768f}; // this is hardcoded, need to
                                                                                        // call arthur function

    // we now get are image only with the 5 values using the threshold :
    // quantizedVesselCorrelation = imquantize(R_VascularPulse - ~maskVesselnessClean * 2, firstThresholds);
    segment_vessels(vesselness_mask_env_.quantizedVesselCorrelation_,
                    vesselness_filter_struct_.thresholds,
                    vesselness_mask_env_.R_vascular_pulse_,
                    vesselness_mask_env_.mask_vesselness_clean_,
                    buffers_.gpu_postprocess_frame_size,
                    thresholds,
                    stream_);
}

#pragma endregion

void Analysis::insert_first_analysis_masks()
{
    LOG_FUNC();

    if (setting<settings::ImageType>() == ImgType::Moments_0 &&
        (setting<settings::ArteryMaskEnabled>() || setting<settings::VeinMaskEnabled>() ||
         setting<settings::ChoroidMaskEnabled>()))
    {
        fn_compute_vect_->push_back(
            [=]()
            {
                // map_multiply(moments_env_.moment0_buffer, 512 * 512, 1.0f / 10000.0f, stream_);
                // map_multiply(moments_env_.moment1_buffer, 512 * 512, 1.0f / 10000.0f, stream_);

                // Reset the mask result to 0
                cudaXMemsetAsync(mask_result_buffer_, 0, sizeof(float) * buffers_.gpu_postprocess_frame_size, stream_);

                // insert_bin_moments();
                compute_pretreatment();
                compute_vesselness_response();
                compute_barycentres_and_circle_mask();
                compute_correlation();
                compute_segment_vessels();
            });
    }
}

void Analysis::insert_artery_mask()
{
    LOG_FUNC();

    if (setting<settings::ImageType>() == ImgType::Moments_0 && setting<settings::ArteryMaskEnabled>())
    {
        fn_compute_vect_->push_back(
            [=]()
            {
                compute_first_mask_artery(buffers_.gpu_postprocess_frame,
                                          vesselness_mask_env_.quantizedVesselCorrelation_,
                                          buffers_.gpu_postprocess_frame_size,
                                          stream_);
                bwareaopen(buffers_.gpu_postprocess_frame,
                           setting<settings::MinMaskArea>(),
                           fd_.width,
                           fd_.height,
                           bw_area_env_.uint_buffer_1_.get(),
                           bw_area_env_.uint_buffer_2_.get(),
                           bw_area_env_.float_buffer_.get(),
                           bw_area_env_.size_t_gpu_.get(),
                           stream_);

                apply_mask_or(buffers_.gpu_postprocess_frame, mask_result_buffer_, fd_.width, fd_.height, stream_);
                cudaXMemcpyAsync(mask_result_buffer_,
                                 buffers_.gpu_postprocess_frame,
                                 buffers_.gpu_postprocess_frame_size * sizeof(float),
                                 cudaMemcpyDeviceToDevice,
                                 stream_);

                shift_corners(buffers_.gpu_postprocess_frame.get(), 1, fd_.width, fd_.height, stream_);
            });
    }
}

void Analysis::insert_vein_mask()
{
    LOG_FUNC();

    if (setting<settings::ImageType>() == ImgType::Moments_0 && setting<settings::VeinMaskEnabled>())
    {
        fn_compute_vect_->push_back(
            [=]()
            {
                compute_first_mask_vein(buffers_.gpu_postprocess_frame,
                                        vesselness_mask_env_.quantizedVesselCorrelation_,
                                        buffers_.gpu_postprocess_frame_size,
                                        stream_);
                bwareaopen(buffers_.gpu_postprocess_frame,
                           setting<settings::MinMaskArea>(),
                           fd_.width,
                           fd_.height,
                           bw_area_env_.uint_buffer_1_.get(),
                           bw_area_env_.uint_buffer_2_.get(),
                           bw_area_env_.float_buffer_.get(),
                           bw_area_env_.size_t_gpu_.get(),
                           stream_);

                apply_mask_or(buffers_.gpu_postprocess_frame, mask_result_buffer_, fd_.width, fd_.height, stream_);
                cudaXMemcpyAsync(mask_result_buffer_,
                                 buffers_.gpu_postprocess_frame,
                                 buffers_.gpu_postprocess_frame_size * sizeof(float),
                                 cudaMemcpyDeviceToDevice,
                                 stream_);

                shift_corners(buffers_.gpu_postprocess_frame.get(), 1, fd_.width, fd_.height, stream_);
            });
    }
}

void Analysis::insert_choroid_mask()
{
    LOG_FUNC();

    if (setting<settings::ImageType>() == ImgType::Moments_0 && setting<settings::ChoroidMaskEnabled>())
    {
        fn_compute_vect_->push_back(
            [=]()
            {
                cudaXMemcpyAsync(first_mask_choroid_struct_.first_mask_choroid,
                                 vesselness_mask_env_.mask_vesselness_clean_,
                                 buffers_.gpu_postprocess_frame_size * sizeof(float),
                                 cudaMemcpyDeviceToDevice,
                                 stream_);

                negation(first_mask_choroid_struct_.first_mask_choroid, buffers_.gpu_postprocess_frame_size, stream_);

                apply_mask_and(first_mask_choroid_struct_.first_mask_choroid,
                               vesselness_mask_env_.mask_vesselness_,
                               fd_.width,
                               fd_.height,
                               stream_);

                compute_first_mask_artery(buffers_.gpu_postprocess_frame,
                                          vesselness_mask_env_.quantizedVesselCorrelation_,
                                          buffers_.gpu_postprocess_frame_size,
                                          stream_);

                negation(buffers_.gpu_postprocess_frame, buffers_.gpu_postprocess_frame_size, stream_);

                apply_mask_and(first_mask_choroid_struct_.first_mask_choroid,
                               buffers_.gpu_postprocess_frame,
                               fd_.width,
                               fd_.height,
                               stream_);

                compute_first_mask_vein(buffers_.gpu_postprocess_frame,
                                        vesselness_mask_env_.quantizedVesselCorrelation_,
                                        buffers_.gpu_postprocess_frame_size,
                                        stream_);

                negation(buffers_.gpu_postprocess_frame, buffers_.gpu_postprocess_frame_size, stream_);

                apply_mask_and(first_mask_choroid_struct_.first_mask_choroid,
                               buffers_.gpu_postprocess_frame,
                               fd_.width,
                               fd_.height,
                               stream_);

                cudaXMemcpyAsync(buffers_.gpu_postprocess_frame,
                                 first_mask_choroid_struct_.first_mask_choroid,
                                 sizeof(float) * buffers_.gpu_postprocess_frame_size,
                                 cudaMemcpyDeviceToDevice,
                                 stream_);
                apply_mask_or(buffers_.gpu_postprocess_frame, mask_result_buffer_, fd_.width, fd_.height, stream_);
                cudaXMemcpyAsync(mask_result_buffer_,
                                 buffers_.gpu_postprocess_frame,
                                 buffers_.gpu_postprocess_frame_size * sizeof(float),
                                 cudaMemcpyDeviceToDevice,
                                 stream_);
                shift_corners(buffers_.gpu_postprocess_frame.get(), 1, fd_.width, fd_.height, stream_);
            });
    }
}

void Analysis::insert_chart()
{
    LOG_FUNC();
    if (setting<settings::ChartMeanVesselsEnabled>() && setting<settings::ImageType>() == ImgType::Moments_0 &&
        (setting<settings::VeinMaskEnabled>() || setting<settings::ArteryMaskEnabled>() ||
         setting<settings::ChoroidMaskEnabled>()))
    {
        fn_compute_vect_->push_back(
            [=]()
            {
                float* mask_buffer = get_mask_result();
                auto points = get_sum_with_mask(moments_env_.moment0_buffer, // TODO change with the correct buffer
                                                vesselness_mask_env_.quantizedVesselCorrelation_,
                                                fd_.get_frame_size(),
                                                chart_mean_vessels_env_.float_buffer_gpu_,
                                                stream_);

                chart_mean_vessels_env_.chart_display_queue_->push_back(points);
            });
    }
}

} // namespace holovibes::analysis