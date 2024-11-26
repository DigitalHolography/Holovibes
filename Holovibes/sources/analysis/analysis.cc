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

#define DIAPHRAGM_FACTOR 0.4f
#define FROM_CSV false

namespace holovibes::analysis
{

void Analysis::init()
{
    LOG_FUNC();
    const size_t frame_res = fd_.get_frame_res();

    // No need for memset here since it will be completely overwritten by
    // cuComplex values
    buffers_.gpu_convolution_buffer.resize(frame_res);

    // No need for memset here since it will be memset in the actual convolution
    cuComplex_buffer_.resize(frame_res);

    // Prepare gaussian blur kernel
    std::vector<float> gaussian_kernel;
    api::load_convolution_matrix_file("gaussian_128_128_1.txt", gaussian_kernel);

    gaussian_128_kernel_buffer_.resize(frame_res);
    cudaXMemsetAsync(gaussian_128_kernel_buffer_.get(), 0, frame_res * sizeof(cuComplex), stream_);
    cudaSafeCall(cudaMemcpy2DAsync(gaussian_128_kernel_buffer_.get(),
                                   sizeof(cuComplex),
                                   gaussian_kernel.data(),
                                   sizeof(float),
                                   sizeof(float),
                                   frame_res,
                                   cudaMemcpyHostToDevice,
                                   stream_));
    cudaXStreamSynchronize(stream_);

    constexpr uint batch_size = 1; // since only one frame.
    // We compute the FFT of the kernel, once, here, instead of every time the
    // convolution subprocess is called

    int err = 0;
    err += !uint_buffer_1_.resize(frame_res);
    err += !uint_buffer_2_.resize(frame_res);
    err += !float_buffer_.resize(frame_res);

    shift_corners(gaussian_128_kernel_buffer_.get(), batch_size, fd_.width, fd_.height, stream_);
    cufftSafeCall(cufftExecC2C(convolution_plan_,
                               gaussian_128_kernel_buffer_.get(),
                               gaussian_128_kernel_buffer_.get(),
                               CUFFT_FORWARD));

    // Init CircularVideoBuffer
    vesselness_mask_env_.m0_ff_video_cb_ =
        std::make_unique<CircularVideoBuffer>(frame_res, api::get_time_window(), stream_);

    vesselness_mask_env_.f_avg_video_cb_ =
        std::make_unique<CircularVideoBuffer>(frame_res, api::get_time_window(), stream_);

    vesselness_mask_env_.vascular_pulse_video_cb_ =
        std::make_unique<CircularVideoBuffer>(frame_res, api::get_time_window(), stream_);

    // (Re)Allocate vesselness buffers, can handle frame size change
    vesselness_mask_env_.time_window_ = api::get_time_window();
    vesselness_mask_env_.number_image_mean_ = 0;

    err += !vesselness_mask_env_.m0_ff_sum_image_.resize(buffers_.gpu_postprocess_frame_size);
    err += !vesselness_mask_env_.image_with_mean_.resize(buffers_.gpu_postprocess_frame_size);
    err += !vesselness_mask_env_.m0_ff_video_centered_.resize(buffers_.gpu_postprocess_frame_size *
                                                              506); // TODO: time_window

    err += !vesselness_mask_env_.vascular_image_.resize(frame_res);

    // Allocate vesselness filter struct internal buffers
    err += !vesselness_filter_struct_.I.resize(frame_res);
    err += !vesselness_filter_struct_.convolution_tmp_buffer.resize(frame_res);
    err += !vesselness_filter_struct_.H.resize(frame_res * 3);
    err += !vesselness_filter_struct_.lambda_1.resize(frame_res);
    err += !vesselness_filter_struct_.lambda_2.resize(frame_res);
    err += !vesselness_filter_struct_.R_blob.resize(frame_res);
    err += !vesselness_filter_struct_.c_temp.resize(frame_res);

    err += !vesselness_mask_env_.m1_divided_by_m0_frame_.resize(frame_res);
    err += !vesselness_mask_env_.circle_mask_.resize(frame_res);
    err += !vesselness_mask_env_.bwareafilt_result_.resize(frame_res);
    err += !vesselness_mask_env_.mask_vesselness_clean_.resize(frame_res);

    // Compute gaussian deriviatives kernels according to simga
    float sigma = setting<settings::VesselnessSigma>();

    int gamma = 1;
    int A = std::pow(sigma, gamma);
    int x_lim = std::ceil(4 * sigma);
    int y_lim = std::ceil(4 * sigma);
    int x_size = x_lim * 2 + 1;
    int y_size = y_lim * 2 + 1;

    vesselness_mask_env_.kernel_x_size_ = x_size;
    vesselness_mask_env_.kernel_y_size_ = y_size;

    // Initialize normalized centered at 0 lists, ex for x_size = 3 : [-1, 0, 1]
    float* x;
    cudaXMalloc(&x, x_size * sizeof(float));
    normalized_list(x, x_lim, x_size, stream_);

    float* y;
    cudaXMalloc(&y, y_size * sizeof(float));
    normalized_list(y, y_lim, y_size, stream_);

    // Initialize X and Y deriviative gaussian kernels
    float* g_xx_px;
    cudaXMalloc(&g_xx_px, x_size * sizeof(float));
    comp_dgaussian(g_xx_px, x, x_size, sigma, 2, stream_);

    float* g_xx_qy;
    cudaXMalloc(&g_xx_qy, y_size * sizeof(float));
    comp_dgaussian(g_xx_qy, y, y_size, sigma, 0, stream_);

    cudaXStreamSynchronize(stream_);

    err += !vesselness_mask_env_.g_xx_mul_.resize(x_size * y_size);
    holovibes::compute::matrix_multiply<float>(g_xx_qy,
                                               g_xx_px,
                                               y_size,
                                               x_size,
                                               1,
                                               vesselness_mask_env_.g_xx_mul_.get(),
                                               cublas_handler_,
                                               CUBLAS_OP_N);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    float* result_transpose;
    cudaXMalloc(&result_transpose, sizeof(float) * x_size * y_size);
    cublasSafeCall(cublasSgeam(cublas_handler_,
                               CUBLAS_OP_T,
                               CUBLAS_OP_N,
                               x_size,
                               y_size,
                               &alpha,
                               vesselness_mask_env_.g_xx_mul_,
                               y_size,
                               &beta,
                               nullptr,
                               y_size,
                               result_transpose,
                               x_size));

    vesselness_mask_env_.g_xx_mul_.reset(result_transpose);

    cudaXFree(g_xx_qy);
    cudaXFree(g_xx_px);

    float* g_xy_px;
    cudaXMalloc(&g_xy_px, x_size * sizeof(float));
    comp_dgaussian(g_xy_px, x, x_size, sigma, 1, stream_);

    float* g_xy_qy;
    cudaXMalloc(&g_xy_qy, y_size * sizeof(float));
    comp_dgaussian(g_xy_qy, y, y_size, sigma, 1, stream_);

    err += !vesselness_mask_env_.g_xy_mul_.resize(x_size * y_size);
    holovibes::compute::matrix_multiply<float>(g_xy_qy,
                                               g_xy_px,
                                               y_size,
                                               x_size,
                                               1,
                                               vesselness_mask_env_.g_xy_mul_.get(),
                                               cublas_handler_);

    cudaXMalloc(&result_transpose, sizeof(float) * x_size * y_size);
    cublasSafeCall(cublasSgeam(cublas_handler_,
                               CUBLAS_OP_T,
                               CUBLAS_OP_N,
                               x_size,
                               y_size,
                               &alpha,
                               vesselness_mask_env_.g_xy_mul_,
                               y_size,
                               &beta,
                               nullptr,
                               y_size,
                               result_transpose,
                               x_size));

    vesselness_mask_env_.g_xy_mul_.reset(result_transpose);

    cudaXFree(g_xy_qy);
    cudaXFree(g_xy_px);

    float* g_yy_px;
    cudaXMalloc(&g_yy_px, x_size * sizeof(float));
    comp_dgaussian(g_yy_px, x, x_size, sigma, 0, stream_);

    float* g_yy_qy;
    cudaXMalloc(&g_yy_qy, y_size * sizeof(float));
    comp_dgaussian(g_yy_qy, x, y_size, sigma, 2, stream_);

    // Compute qy * px matrices to simply two 1D convolutions to one 2D convolution
    err += !vesselness_mask_env_.g_yy_mul_.resize(x_size * y_size);
    holovibes::compute::matrix_multiply<float>(g_yy_qy,
                                               g_yy_px,
                                               y_size,
                                               x_size,
                                               1,
                                               vesselness_mask_env_.g_yy_mul_.get(),
                                               cublas_handler_);

    cudaXMalloc(&result_transpose, sizeof(float) * x_size * y_size);
    cublasSafeCall(cublasSgeam(cublas_handler_,
                               CUBLAS_OP_T,
                               CUBLAS_OP_N,
                               x_size,
                               y_size,
                               &alpha,
                               vesselness_mask_env_.g_yy_mul_,
                               y_size,
                               &beta,
                               nullptr,
                               y_size,
                               result_transpose,
                               x_size));

    vesselness_mask_env_.g_yy_mul_.reset(result_transpose);

    cudaXFree(g_yy_qy);
    cudaXFree(g_yy_px);

    // Compute Gaussian kernel for vascular pulse
    float sigma_2 = 0.02 * fd_.width;
    vesselness_mask_env_.vascular_kernel_size_ = 2 * std::ceil(2 * sigma_2) + 1;

    err += !vesselness_mask_env_.vascular_kernel_.resize(vesselness_mask_env_.vascular_kernel_size_ *
                                                         vesselness_mask_env_.vascular_kernel_size_);
    compute_gauss_kernel(vesselness_mask_env_.vascular_kernel_, sigma_2, stream_);

    if (err != 0)
        throw std::exception(cudaGetErrorString(cudaGetLastError()));
}

void Analysis::dispose()
{
    LOG_FUNC();

    buffers_.gpu_convolution_buffer.reset(nullptr);

    gaussian_128_kernel_buffer_.reset();
    cuComplex_buffer_.reset();

    vesselness_mask_env_.g_xx_mul_.reset();
    vesselness_mask_env_.g_xy_mul_.reset();
    vesselness_mask_env_.g_yy_mul_.reset();

    uint_buffer_1_.reset();
    uint_buffer_2_.reset();
    float_buffer_.reset();
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

                // convolution_kernel(buffers_.gpu_postprocess_frame,
                //                    buffers_.gpu_convolution_buffer,
                //                    cuComplex_buffer_.get(),
                //                    &convolution_plan_,
                //                    fd_.get_frame_res(),
                //                    gaussian_128_kernel_buffer_.get(),
                //                    false,
                //                    stream_);

                // Fill the m0_ff_video circular buffer with the flat field corrected images and compute the mean from
                // it
                vesselness_mask_env_.m0_ff_video_cb_->add_new_frame(buffers_.gpu_postprocess_frame);
                vesselness_mask_env_.m0_ff_video_cb_->compute_mean_image();

                // Compute the centered image from the temporal mean of the video
                image_centering(vesselness_mask_env_.m0_ff_video_centered_,
                                vesselness_mask_env_.m0_ff_video_cb_->get_data_ptr(),
#if FROM_CSV
                                m0_ff_img_csv_,
#else
                                vesselness_mask_env_.m0_ff_video_cb_->get_mean_image(),
#endif
                                buffers_.gpu_postprocess_frame_size,
                                i_,
                                stream_);

                // Compute the first vesselness mask which represents both vessels types (arteries and veins)
                vesselness_filter(buffers_.gpu_postprocess_frame,
#if FROM_CSV
                                  m0_ff_img_csv_,
#else
                                  vesselness_mask_env_.m0_ff_video_cb_->get_mean_image(),
#endif
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
#if !FROM_CSV
                // Binarize the vesselness output to produce the mask vesselness
                compute_binarise_otsu(buffers_.gpu_postprocess_frame, fd_.width, fd_.height, stream_);

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
#endif

                // Compute vascular image
                compute_multiplication(vesselness_mask_env_.vascular_image_,
#if FROM_CSV
                                       m0_ff_img_csv_,
                                       f_avg_csv_,
#else
                                       vesselness_mask_env_.m0_ff_video_cb_->get_mean_image(),
                                       vesselness_mask_env_.f_avg_video_cb_->get_mean_image(),
#endif
                                       buffers_.gpu_postprocess_frame_size,
                                       1,
                                       stream_);
                // From here ~115 FPS

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

                // From here ~100 FPS

                int CRV_index = compute_barycentre_circle_mask(vesselness_mask_env_.circle_mask_,
                                                               vesselness_mask_env_.vascular_image_,
                                                               buffers_.gpu_postprocess_frame_size,
                                                               stream_);

                cudaXMemcpyAsync(vesselness_mask_env_.bwareafilt_result_,
#if FROM_CSV
                                 mask_vesselness_csv_,
#else
                                 buffers_.gpu_postprocess_frame, // This is the result of otsu
#endif
                                 sizeof(float) * buffers_.gpu_postprocess_frame_size,
                                 cudaMemcpyDeviceToDevice,
                                 stream_);

                // apply_mask_or(vesselness_mask_env_.bwareafilt_result_, vesselness_mask_env_.circle_mask_, fd_.width,
                // fd_.height, stream_);

                // bwareafilt(vesselness_mask_env_.bwareafilt_result_,
                //            fd_.width,
                //            fd_.height,
                //            uint_buffer_1_,
                //            uint_buffer_2_,
                //            float_buffer_,
                //            cublas_handler_,
                //            stream_);

                // TODO: get titouan to fix his function and remove csv
                cudaXMemcpyAsync(vesselness_mask_env_.bwareafilt_result_,
                                 bwareafilt_csv_,
                                 512 * 512 * sizeof(float),
                                 cudaMemcpyDeviceToDevice,
                                 stream_);

                cudaXMemcpyAsync(vesselness_mask_env_.mask_vesselness_clean_,
#if FROM_CSV

                                 mask_vesselness_csv_,
#else
                                 buffers_.gpu_postprocess_frame,
#endif
                                 sizeof(float) * buffers_.gpu_postprocess_frame_size,
                                 cudaMemcpyDeviceToDevice,
                                 stream_);

                apply_mask_and(vesselness_mask_env_.mask_vesselness_clean_,
                               vesselness_mask_env_.bwareafilt_result_,
                               fd_.width,
                               fd_.height,
                               stream_);

                cudaXMemcpyAsync(vesselness_mask_env_.mask_vesselness_clean_,
                                 mask_vesselness_clean_csv_,
                                 512 * 512 * sizeof(float),
                                 cudaMemcpyDeviceToDevice,
                                 stream_);
                cudaXStreamSynchronize(stream_);

                vesselness_mask_env_.m0_ff_video_cb_->multiply_data_by_frame(
                    vesselness_mask_env_.mask_vesselness_clean_);

                float* vascular_pulse;
                cudaXMalloc(&vascular_pulse, 506 * sizeof(float));
                cudaXMemcpy(vascular_pulse,
                            vesselness_mask_env_.m0_ff_video_cb_->get_mean_1_2_(),
                            i_ * sizeof(float),
                            cudaMemcpyDeviceToDevice);

                int nnz = count_non_zero(vesselness_mask_env_.mask_vesselness_clean_, fd_.height, fd_.width, stream_);
                compute_first_correlation(buffers_.gpu_postprocess_frame, // R_vascular_pulse will be
                                                                          // in this buffer
                                          vesselness_mask_env_.m0_ff_video_centered_,
#if FROM_CSV

                                          vascular_pulse,
                                          11727,
                                          506,
#else
                                          vascular_pulse,
                                          nnz,
                                          vesselness_mask_env_.m0_ff_video_cb_->get_frame_count(),
#endif
                                          buffers_.gpu_postprocess_frame_size,
                                          stream_);
                cudaXFree(vascular_pulse);

                // multiply_three_vectors(vesselness_mask_env_.vascular_image_,
                // #if FROM_CSV
                //                                        m0_ff_img_csv_,
                //                                        f_avg_csv_,
                //                                        R_VascularPulse_csv_,
                // #else
                //                                        vesselness_mask_env_.m0_ff_video_centered_,
                //                                        vesselness_mask_env_.f_avg_video_cb_->get_mean_image(),
                //                                        buffers_.gpu_postprocess_frame,
                // #endif
                //                                        buffers_.gpu_postprocess_frame_size,
                //                                        stream_);
                //                 cudaXMemcpyAsync(buffers_.gpu_postprocess_frame,
                //                                  vesselness_mask_env_.vascular_image_,
                //                                  sizeof(float) * 512 * 512,
                //                                  cudaMemcpyDeviceToDevice,
                //                                  stream_);
                //                 apply_convolution(vesselness_mask_env_.vascular_image_,
                //                                   vesselness_mask_env_.vascular_kernel_,
                //                                   fd_.width,
                //                                   fd_.height,
                //                                   vesselness_mask_env_.vascular_kernel_size_,
                //                                   vesselness_mask_env_.vascular_kernel_size_,
                //                                   vesselness_filter_struct_.convolution_tmp_buffer,
                //                                   stream_,
                //                                   ConvolutionPaddingType::SCALAR,
                //                                   0);
                //                 apply_diaphragm_mask(vesselness_mask_env_.vascular_image_,
                //                                      fd_.width / 2 - 1,
                //                                      fd_.height / 2 - 1,
                //                                      DIAPHRAGM_FACTOR * (fd_.width + fd_.height) / 2,
                //                                      fd_.width,
                //                                      fd_.height,
                //                                      stream_);

                //                 compute_barycentre_circle_mask(vesselness_mask_env_.circle_mask_,
                //                                                vesselness_mask_env_.vascular_image_,
                //                                                buffers_.gpu_postprocess_frame_size,
                //                                                stream_,
                //                                                CRV_index);

                //                 cudaXMalloc(&(vesselness_mask_env_.quantizedVesselCorrelation_),
                //                             sizeof(float) * buffers_.gpu_postprocess_frame_size);
                // #if FROM_CSV
                //                 segment_vessels(vesselness_mask_env_.quantizedVesselCorrelation_,
                //                                 R_VascularPulse_csv_,
                //                                 mask_vesselness_clean_csv_,
                //                                 buffers_.gpu_postprocess_frame_size,
                //                                 stream_);
                // #else
                //             // TODO
                // #endif
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
                // compute_first_mask_artery(buffers_.gpu_postprocess_frame,
                //                           vesselness_mask_env_.quantizedVesselCorrelation_,
                //                           buffers_.gpu_postprocess_frame_size,
                //                           stream_);
                // cudaXFree(vesselness_mask_env_.quantizedVesselCorrelation_);
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
                cudaXStreamSynchronize(stream_);
                cudaXFree(vesselness_mask_env_.quantizedVesselCorrelation_);
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
                cudaXStreamSynchronize(stream_);
                cudaXFree(vesselness_mask_env_.quantizedVesselCorrelation_);
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
                float* first_mask_choroid;
                cudaXMalloc(&first_mask_choroid, sizeof(float) * buffers_.gpu_postprocess_frame_size);
                cudaXMemcpy(first_mask_choroid,
                            mask_vesselness_clean_csv_,
                            buffers_.gpu_postprocess_frame_size * sizeof(float),
                            cudaMemcpyDeviceToDevice);

                negation(first_mask_choroid, buffers_.gpu_postprocess_frame_size, stream_);

                apply_mask_and(first_mask_choroid, mask_vesselness_csv_, fd_.width, fd_.height, stream_);

                compute_first_mask_artery(buffers_.gpu_postprocess_frame,
                                          vesselness_mask_env_.quantizedVesselCorrelation_,
                                          buffers_.gpu_postprocess_frame_size,
                                          stream_);

                negation(buffers_.gpu_postprocess_frame, buffers_.gpu_postprocess_frame_size, stream_);

                apply_mask_and(first_mask_choroid, buffers_.gpu_postprocess_frame, fd_.width, fd_.height, stream_);

                compute_first_mask_vein(buffers_.gpu_postprocess_frame,
                                        vesselness_mask_env_.quantizedVesselCorrelation_,
                                        buffers_.gpu_postprocess_frame_size,
                                        stream_);

                negation(buffers_.gpu_postprocess_frame, buffers_.gpu_postprocess_frame_size, stream_);

                apply_mask_and(first_mask_choroid, buffers_.gpu_postprocess_frame, fd_.width, fd_.height, stream_);

                cudaXMemcpy(buffers_.gpu_postprocess_frame,
                            first_mask_choroid,
                            sizeof(float) * buffers_.gpu_postprocess_frame_size,
                            cudaMemcpyDeviceToDevice);
                shift_corners(buffers_.gpu_postprocess_frame.get(), 1, fd_.width, fd_.height, stream_);

                cudaXFree(first_mask_choroid);
                cudaXFree(vesselness_mask_env_.quantizedVesselCorrelation_);
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
                // cublasHandle_t& handle = cuda_tools::CublasHandle::instance();
                // int maxI = -1;
                // int minI = -1;
                // cublasIsamax(handle, buffers_.gpu_postprocess_frame_size, buffers_.gpu_postprocess_frame, 1, &maxI);
                // cublasIsamin(handle, buffers_.gpu_postprocess_frame_size, buffers_.gpu_postprocess_frame, 1, &minI);

                // float h_min, h_max;
                // cudaXMemcpy(&h_min, buffers_.gpu_postprocess_frame + (minI - 1), sizeof(float),
                // cudaMemcpyDeviceToHost); cudaXMemcpy(&h_max, buffers_.gpu_postprocess_frame + (maxI - 1),
                // sizeof(float), cudaMemcpyDeviceToHost);

                // normalise(buffers_.gpu_postprocess_frame, h_min, h_max, buffers_.gpu_postprocess_frame_size,
                // stream_);

                // print_in_file_gpu(buffers_.gpu_postprocess_frame, 512, 512, "before_otsu_normalized", stream_);

                if (setting<settings::OtsuKind>() == OtsuKind::Adaptive)
                {

                    float* d_output;
                    cudaMalloc(&d_output, buffers_.gpu_postprocess_frame_size * sizeof(float));
                    compute_binarise_otsu_bradley(buffers_.gpu_postprocess_frame,
                                                  d_output,
                                                  fd_.width,
                                                  fd_.height,
                                                  setting<settings::OtsuWindowSize>(),
                                                  setting<settings::OtsuLocalThreshold>(),
                                                  stream_);

                    cudaXMemcpy(buffers_.gpu_postprocess_frame,
                                d_output,
                                buffers_.gpu_postprocess_frame_size * sizeof(float),
                                cudaMemcpyDeviceToDevice);
                    cudaFree(d_output);
                }
                else
                    compute_binarise_otsu(buffers_.gpu_postprocess_frame, fd_.width, fd_.height, stream_);
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
            {
                float* image_d = buffers_.gpu_postprocess_frame.get();
                uint* labels_d = uint_buffer_1_.get();
                uint* linked_d = uint_buffer_2_.get();
                float* labels_sizes_d = float_buffer_.get();

                cublasHandle_t& handle = cuda_tools::CublasHandle::instance();

                get_connected_component(labels_d, labels_sizes_d, linked_d, image_d, fd_.width, fd_.height, stream_);

                int maxI = -1;
                cublasIsamax(handle, buffers_.gpu_postprocess_frame_size, labels_sizes_d, 1, &maxI);
                if (maxI - 1 > 0)
                    area_filter(image_d, labels_d, buffers_.gpu_postprocess_frame_size, maxI - 1, stream_);
            }
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
                float* image_d = buffers_.gpu_postprocess_frame.get();
                uint* labels_d = uint_buffer_1_.get();
                uint* linked_d = uint_buffer_2_.get();
                float* labels_sizes_d = float_buffer_.get();

                uint p = setting<settings::MinMaskArea>();

                get_connected_component(labels_d, labels_sizes_d, linked_d, image_d, fd_.width, fd_.height, stream_);
                if (p != 0)
                    area_open(image_d, labels_d, labels_sizes_d, buffers_.gpu_postprocess_frame_size, p, stream_);
            }
        });
}

} // namespace holovibes::analysis
