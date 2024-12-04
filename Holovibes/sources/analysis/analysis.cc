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
#include "imbinarize.cuh"

#define DIAPHRAGM_FACTOR 0.4f
#define OTSU_BINS 256

namespace holovibes::analysis
{

void Analysis::init_params_vesselness_filter(float* result_transpose,
                                             float* target,
                                             float sigma,
                                             int x_size,
                                             int y_size,
                                             int x_lim,
                                             int y_lim,
                                             int p,
                                             int q,
                                             cudaStream_t stream)
{
    // Initialize normalized centered at 0 lists, ex for x_size = 3 : [-1, 0, 1]
    float* x;
    cudaXMalloc(&x, x_size * sizeof(float));
    normalized_list(x, x_lim, x_size, stream);

    float* y;
    cudaXMalloc(&y, y_size * sizeof(float));
    normalized_list(y, y_lim, y_size, stream);

    // Initialize X and Y deriviative gaussian kernels
    float* g_px;
    cudaXMalloc(&g_px, x_size * sizeof(float));
    comp_dgaussian(g_px, x, x_size, sigma, p, stream);

    float* g_qy;
    cudaXMalloc(&g_qy, y_size * sizeof(float));
    comp_dgaussian(g_qy, y, y_size, sigma, q, stream);

    holovibes::compute::matrix_multiply<float>(g_qy, g_px, y_size, x_size, 1, target, cublas_handler_);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSafeCall(cublasSgeam(cublas_handler_,
                               CUBLAS_OP_T,
                               CUBLAS_OP_N,
                               x_size,
                               y_size,
                               &alpha,
                               target,
                               y_size,
                               &beta,
                               nullptr,
                               y_size,
                               result_transpose,
                               x_size));

    // We no longer need those
    cudaXFree(x);
    cudaXFree(y);
    cudaXFree(g_qy);
    cudaXFree(g_px);
}

void Analysis::init()
{
    const size_t frame_res = fd_.get_frame_res();

    // Compute variables for gaussian deriviatives kernels
    float GdkSigma = setting<settings::VesselnessSigma>();
    int gamma = 1;
    int A = std::pow(GdkSigma, gamma);
    int x_lim = std::ceil(4 * GdkSigma);
    int y_lim = std::ceil(4 * GdkSigma);
    int x_size = x_lim * 2 + 1;
    int y_size = y_lim * 2 + 1;

    // Compute Gaussian kernel variable for vascular pulse
    float VpSigma = 0.02 * fd_.width;

    // Used in bwareafilt and bwareaopen
    uint_buffer_1_.safe_resize(frame_res);
    uint_buffer_2_.safe_resize(frame_res);
    size_t_gpu_.resize(1);
    float_buffer_.safe_resize(frame_res);
    otsu_histo_buffer_.resize(OTSU_BINS);
    otsu_float_gpu_.resize(1);
    otsu_histo_buffer_2_.resize(OTSU_BINS);

    // Allocate vesselness mask env buffers
    vesselness_mask_env_.time_window_ = api::get_time_window();
    vesselness_mask_env_.m0_ff_video_centered_.safe_resize(buffers_.gpu_postprocess_frame_size *
                                                           vesselness_mask_env_.time_window_);
    vesselness_mask_env_.vascular_image_.safe_resize(frame_res);
    vesselness_mask_env_.m1_divided_by_m0_frame_.safe_resize(frame_res);
    vesselness_mask_env_.circle_mask_.safe_resize(frame_res);
    vesselness_mask_env_.bwareafilt_result_.safe_resize(frame_res);
    vesselness_mask_env_.mask_vesselness_.safe_resize(frame_res);
    vesselness_mask_env_.mask_vesselness_clean_.safe_resize(frame_res);
    vesselness_mask_env_.quantizedVesselCorrelation_.safe_resize(frame_res);
    vesselness_mask_env_.kernel_x_size_ = x_size;
    vesselness_mask_env_.kernel_y_size_ = y_size;
    vesselness_mask_env_.vascular_kernel_size_ = 2 * std::ceil(2 * VpSigma) + 1;
    vesselness_mask_env_.g_xx_mul_.safe_resize(x_size * y_size);
    vesselness_mask_env_.g_xy_mul_.safe_resize(x_size * y_size);
    vesselness_mask_env_.g_yy_mul_.safe_resize(x_size * y_size);
    vesselness_mask_env_.vascular_kernel_.safe_resize(vesselness_mask_env_.vascular_kernel_size_ *
                                                      vesselness_mask_env_.vascular_kernel_size_);

    vesselness_mask_env_.before_threshold.safe_resize(frame_res);
    vesselness_mask_env_.R_vascular_pulse_.safe_resize(frame_res);
    // Init CircularVideoBuffer
    vesselness_mask_env_.m0_ff_video_cb_ =
        std::make_unique<CircularVideoBuffer>(frame_res, api::get_time_window(), stream_);

    vesselness_mask_env_.f_avg_video_cb_ =
        std::make_unique<CircularVideoBuffer>(frame_res, api::get_time_window(), stream_);

    // Allocate vesselness filter struct buffers
    vesselness_filter_struct_.I.safe_resize(frame_res);
    vesselness_filter_struct_.convolution_tmp_buffer.safe_resize(frame_res);
    vesselness_filter_struct_.H.safe_resize(frame_res * 3);
    vesselness_filter_struct_.lambda_1.safe_resize(frame_res);
    vesselness_filter_struct_.lambda_2.safe_resize(frame_res);
    vesselness_filter_struct_.R_blob.safe_resize(frame_res);
    vesselness_filter_struct_.c_temp.safe_resize(frame_res);
    vesselness_filter_struct_.CRV_circle_mask.safe_resize(frame_res);
    vesselness_filter_struct_.vascular_pulse.safe_resize(vesselness_mask_env_.time_window_);
    vesselness_filter_struct_.vascular_pulse_centered.safe_resize(vesselness_mask_env_.time_window_);
    vesselness_filter_struct_.std_M0_ff_video_centered.safe_resize(buffers_.gpu_postprocess_frame_size);
    vesselness_filter_struct_.std_vascular_pulse_centered.safe_resize(1);
    vesselness_filter_struct_.thresholds.safe_resize(4);

    // Allocate first mask choroid struct buffers
    first_mask_choroid_struct_.first_mask_choroid.safe_resize(frame_res);

    // TODO taille constante a bouger dans le constructeur peut etre
    // Allocate temporary Otsu buffers
    otsu_struct_.d_counts.safe_resize(OTSU_BINS);
    otsu_struct_.d_counts_sum.safe_resize(1);
    otsu_struct_.p.safe_resize(OTSU_BINS);
    otsu_struct_.p_.safe_resize(OTSU_BINS);
    otsu_struct_.sigma_b_squared.safe_resize(OTSU_BINS);
    otsu_struct_.d_mu_tt.safe_resize(1);
    otsu_struct_.d_mu.safe_resize(OTSU_BINS);
    otsu_struct_.d_omega.safe_resize(OTSU_BINS);
    otsu_struct_.is_max.safe_resize(OTSU_BINS);
    // Init gaussian kernels
    float* result_transpose;
    cudaXMalloc(&result_transpose, sizeof(float) * x_size * y_size);

    init_params_vesselness_filter(result_transpose,
                                  vesselness_mask_env_.g_xx_mul_,
                                  GdkSigma,
                                  x_size,
                                  y_size,
                                  x_lim,
                                  y_lim,
                                  2,
                                  0,
                                  stream_);
    vesselness_mask_env_.g_xx_mul_.reset(result_transpose);

    cudaXMalloc(&result_transpose, sizeof(float) * x_size * y_size);
    init_params_vesselness_filter(result_transpose,
                                  vesselness_mask_env_.g_xy_mul_,
                                  GdkSigma,
                                  x_size,
                                  y_size,
                                  x_lim,
                                  y_lim,
                                  1,
                                  1,
                                  stream_);
    vesselness_mask_env_.g_xy_mul_.reset(result_transpose);

    cudaXMalloc(&result_transpose, sizeof(float) * x_size * y_size);
    init_params_vesselness_filter(result_transpose,
                                  vesselness_mask_env_.g_yy_mul_,
                                  GdkSigma,
                                  x_size,
                                  y_size,
                                  x_lim,
                                  y_lim,
                                  0,
                                  2,
                                  stream_);
    vesselness_mask_env_.g_yy_mul_.reset(result_transpose);

    // Compute Gaussian kernel for vascular pulse
    compute_gauss_kernel(vesselness_mask_env_.vascular_kernel_, VpSigma, stream_);
}

void Analysis::insert_first_analysis_masks()
{
    LOG_FUNC();

    if (setting<settings::ImageType>() == ImgType::Moments_0 &&
        (setting<settings::ArteryMaskEnabled>() || setting<settings::VeinMaskEnabled>()))
    {
        fn_compute_vect_->conditional_push_back(
            [=]()
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
                                 sizeof(float) * 512 * 512,
                                 cudaMemcpyDeviceToDevice,
                                 stream_);

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

                // Compute and apply a circular diaphragm mask on the vesselness output
                apply_diaphragm_mask(buffers_.gpu_postprocess_frame,
                                     fd_.width / 2 - 1,
                                     fd_.height / 2 - 1,
                                     DIAPHRAGM_FACTOR * (fd_.width + fd_.height) / 2,
                                     fd_.width,
                                     fd_.height,
                                     stream_);

                // From here ~160 FPS

                // Otsu is unoptimized (~100 FPS after) TODO: merge titouan's otsu
                float threshold = otsu_compute_threshold(buffers_.gpu_postprocess_frame,
                                                         otsu_histo_buffer_2_,
                                                         buffers_.gpu_postprocess_frame_size,
                                                         otsu_struct_,
                                                         stream_);

                // Binarize the vesselness output to produce the mask vesselness
                apply_binarisation(buffers_.gpu_postprocess_frame, threshold, fd_.width, fd_.height, stream_);

                // Store mask_vesselness for later computations
                cudaXMemcpyAsync(vesselness_mask_env_.mask_vesselness_,
                                 buffers_.gpu_postprocess_frame,
                                 sizeof(float) * buffers_.gpu_postprocess_frame_size,
                                 cudaMemcpyDeviceToDevice,
                                 stream_);

                // Compute f_AVG_mean, which is the temporal average of M1 / M0
                cudaXMemcpyAsync(vesselness_mask_env_.m1_divided_by_m0_frame_,
                                 moments_env_.moment1_buffer,
                                 buffers_.gpu_postprocess_frame_size * sizeof(float),
                                 cudaMemcpyDeviceToDevice,
                                 stream_);

                divide_frames_inplace(vesselness_mask_env_.m1_divided_by_m0_frame_,
                                      moments_env_.moment0_buffer,
                                      buffers_.gpu_postprocess_frame_size,
                                      stream_);

                vesselness_mask_env_.f_avg_video_cb_->add_new_frame(vesselness_mask_env_.m1_divided_by_m0_frame_);

                vesselness_mask_env_.f_avg_video_cb_->compute_mean_image();

                // From here ~111 FPS (might change when otsu fix)
                // Compute vascular image
                compute_multiplication(vesselness_mask_env_.vascular_image_,
                                       vesselness_mask_env_.m0_ff_video_cb_->get_mean_image(),
                                       vesselness_mask_env_.f_avg_video_cb_->get_mean_image(),
                                       buffers_.gpu_postprocess_frame_size,
                                       1,
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

                int CRV_index = compute_barycentre_circle_mask(vesselness_mask_env_.circle_mask_,
                                                               vesselness_filter_struct_.CRV_circle_mask,
                                                               vesselness_mask_env_.vascular_image_,
                                                               fd_.width,
                                                               fd_.height,
                                                               stream_);

                // From here ~90 FPS
                // From here, also issue to get good fps because we don't have Titouan's code
                cudaXMemcpyAsync(vesselness_mask_env_.bwareafilt_result_,
                                 buffers_.gpu_postprocess_frame, // This is the result of otsu
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
                           uint_buffer_1_,
                           uint_buffer_2_,
                           float_buffer_,
                           size_t_gpu_,
                           cublas_handler_,
                           stream_);

                cudaXMemcpyAsync(vesselness_mask_env_.mask_vesselness_clean_,
                                 buffers_.gpu_postprocess_frame, // this is the result of otsu
                                 sizeof(float) * buffers_.gpu_postprocess_frame_size,
                                 cudaMemcpyDeviceToDevice,
                                 stream_);

                apply_mask_and(vesselness_mask_env_.mask_vesselness_clean_,
                               vesselness_mask_env_.bwareafilt_result_,
                               fd_.width,
                               fd_.height,
                               stream_);

                // Fps here ~90 FPS
                vesselness_mask_env_.m0_ff_video_cb_->compute_mean_1_2(vesselness_mask_env_.mask_vesselness_clean_);
                // Fps here ~45 FPS, better than before (5 FPS)
                // With all the other code, we are at 40 FPS so we won't test

                cudaXMemcpy(vesselness_filter_struct_.vascular_pulse,
                            vesselness_mask_env_.m0_ff_video_cb_->get_mean_1_2_(),
                            vesselness_mask_env_.m0_ff_video_cb_->get_frame_count() * sizeof(float),
                            cudaMemcpyDeviceToDevice);

                int nnz = count_non_zero(vesselness_mask_env_.mask_vesselness_clean_, fd_.height, fd_.width, stream_);
                compute_first_correlation(vesselness_mask_env_.R_vascular_pulse_,
                                          vesselness_mask_env_.m0_ff_video_centered_,
                                          vesselness_filter_struct_.vascular_pulse,
                                          nnz,
                                          vesselness_mask_env_.m0_ff_video_cb_->get_frame_count(),
                                          vesselness_filter_struct_,
                                          buffers_.gpu_postprocess_frame_size,
                                          stream_);

                multiply_three_vectors(vesselness_mask_env_.vascular_image_,
                                       vesselness_mask_env_.m0_ff_video_cb_->get_mean_image(),
                                       vesselness_mask_env_.f_avg_video_cb_->get_mean_image(),
                                       vesselness_mask_env_.R_vascular_pulse_,
                                       buffers_.gpu_postprocess_frame_size,
                                       stream_);

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

                apply_diaphragm_mask(vesselness_mask_env_.vascular_image_,
                                     fd_.width / 2 - 1,
                                     fd_.height / 2 - 1,
                                     DIAPHRAGM_FACTOR * (fd_.width + fd_.height) / 2,
                                     fd_.width,
                                     fd_.height,
                                     stream_);

                float thresholds[3] = {0.207108953480839f,
                                       0.334478400506137f,
                                       0.458741275652768f}; // this is hardcoded, need to call arthur function

                segment_vessels(vesselness_mask_env_.quantizedVesselCorrelation_,
                                vesselness_filter_struct_.thresholds,
                                vesselness_mask_env_.R_vascular_pulse_,
                                vesselness_mask_env_.mask_vesselness_clean_,
                                buffers_.gpu_postprocess_frame_size,
                                thresholds,
                                stream_);

                ///////////////////////////////
                // Everything is OK before here
                ///////////////////////////////
            });
    }
}

void Analysis::insert_artery_mask()
{
    LOG_FUNC();

    if (setting<settings::ImageType>() == ImgType::Moments_0 && setting<settings::ArteryMaskEnabled>() &&
        !setting<settings::VeinMaskEnabled>())
    {
        fn_compute_vect_->conditional_push_back(
            [=]()
            {
                compute_first_mask_artery(buffers_.gpu_postprocess_frame,
                                          vesselness_mask_env_.quantizedVesselCorrelation_,
                                          buffers_.gpu_postprocess_frame_size,
                                          stream_);
                // bwareaopen is 4 neighbours currently, should be 8
                bwareaopen(buffers_.gpu_postprocess_frame,
                           150,
                           fd_.width,
                           fd_.height,
                           uint_buffer_1_.get(),
                           uint_buffer_2_.get(),
                           float_buffer_.get(),
                           size_t_gpu_.get(),
                           stream_);
                shift_corners(buffers_.gpu_postprocess_frame.get(), 1, fd_.width, fd_.height, stream_);
            });
    }
}

void Analysis::insert_vein_mask()
{
    LOG_FUNC();

    if (setting<settings::ImageType>() == ImgType::Moments_0 && setting<settings::VeinMaskEnabled>() &&
        !setting<settings::ArteryMaskEnabled>())
    {
        fn_compute_vect_->conditional_push_back(
            [=]()
            {
                compute_first_mask_vein(buffers_.gpu_postprocess_frame,
                                        vesselness_mask_env_.quantizedVesselCorrelation_,
                                        buffers_.gpu_postprocess_frame_size,
                                        stream_);
                bwareaopen(buffers_.gpu_postprocess_frame,
                           150,
                           fd_.width,
                           fd_.height,
                           uint_buffer_1_.get(),
                           uint_buffer_2_.get(),
                           float_buffer_.get(),
                           size_t_gpu_.get(),
                           stream_);
                shift_corners(buffers_.gpu_postprocess_frame.get(), 1, fd_.width, fd_.height, stream_);
            });
    }
}

void Analysis::insert_vesselness()
{
    LOG_FUNC();

    if (setting<settings::ImageType>() == ImgType::Moments_0 && setting<settings::VeinMaskEnabled>() &&
        setting<settings::ArteryMaskEnabled>())
    {
        fn_compute_vect_->conditional_push_back(
            [=]()
            {
                cudaXMemcpy(buffers_.gpu_postprocess_frame,
                            vesselness_mask_env_.quantizedVesselCorrelation_,
                            buffers_.gpu_postprocess_frame_size * sizeof(float),
                            cudaMemcpyDeviceToDevice);
                shift_corners(buffers_.gpu_postprocess_frame.get(), 1, fd_.width, fd_.height, stream_);
            });
    }
}

void Analysis::insert_choroid_mask()
{
    LOG_FUNC();

    if (setting<settings::ImageType>() == ImgType::Moments_0 && setting<settings::VeinMaskEnabled>() &&
        setting<settings::ArteryMaskEnabled>())
    {
        fn_compute_vect_->conditional_push_back(
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
                shift_corners(buffers_.gpu_postprocess_frame.get(), 1, fd_.width, fd_.height, stream_);
            });
    }
}

void Analysis::insert_otsu()
{
    LOG_FUNC();

    if (setting<settings::ImageType>() == ImgType::Moments_0 && setting<settings::OtsuEnabled>() == true)
    {

        fn_compute_vect_->conditional_push_back(
            [=]()
            {
                // if (setting<settings::OtsuKind>() == OtsuKind::Adaptive)
                // {
                //     compute_binarise_otsu_bradley(float_buffer_.get(),
                //                                   otsu_histo_buffer_.get(),
                //                                   buffers_.gpu_postprocess_frame,
                //                                   otsu_float_gpu_.get(),
                //                                   fd_.width,
                //                                   fd_.height,
                //                                   setting<settings::OtsuWindowSize>(),
                //                                   setting<settings::OtsuLocalThreshold>(),
                //                                   stream_);
                //     cudaXMemcpy(buffers_.gpu_postprocess_frame,
                //                 float_buffer_.get(),
                //                 buffers_.gpu_postprocess_frame_size * sizeof(float),
                //                 cudaMemcpyDeviceToDevice);
                // }
                // else
                //     compute_binarise_otsu(buffers_.gpu_postprocess_frame,
                //                           otsu_histo_buffer_.get(),
                //                           otsu_float_gpu_.get(),
                //                           fd_.width,
                //                           fd_.height,
                //                           stream_);
            });
    }
}

void Analysis::insert_bwareafilt()
{
    LOG_FUNC();

    fn_compute_vect_->conditional_push_back(
        [=]()
        {
            if (setting<settings::ImageType>() == ImgType::Moments_0 && setting<settings::BwareafiltEnabled>() == true)
                bwareafilt(buffers_.gpu_postprocess_frame.get(),
                           fd_.width,
                           fd_.height,
                           uint_buffer_1_.get(),
                           uint_buffer_2_.get(),
                           float_buffer_.get(),
                           size_t_gpu_.get(),
                           cuda_tools::CublasHandle::instance(),
                           stream_);
        });
}

void Analysis::insert_bwareaopen()
{
    LOG_FUNC();

    fn_compute_vect_->conditional_push_back(
        [=]()
        {
            if (setting<settings::ImageType>() == ImgType::Moments_0 && setting<settings::BwareaopenEnabled>() == true)
            {
                shift_corners(buffers_.gpu_postprocess_frame.get(), 1, fd_.width, fd_.height, stream_);
                bwareaopen(buffers_.gpu_postprocess_frame.get(),
                           setting<settings::MinMaskArea>(),
                           fd_.width,
                           fd_.height,
                           uint_buffer_1_.get(),
                           uint_buffer_2_.get(),
                           float_buffer_.get(),
                           size_t_gpu_.get(),
                           stream_);
            }
        });
}

} // namespace holovibes::analysis