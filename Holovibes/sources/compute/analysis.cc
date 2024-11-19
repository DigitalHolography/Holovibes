#include "analysis.hh"
#include "icompute.hh"

#include "convolution.cuh"
#include "tools.cuh"
#include "tools_compute.cuh"
#include "contrast_correction.cuh"
#include "cuda_memory.cuh"
#include "shift_corners.cuh"
#include "map.cuh"
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

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>

using holovibes::cuda_tools::CufftHandle;

#define DIAPHRAGM_FACTOR 0.4f
#define FROM_CSV true

namespace holovibes::compute
{

void Analysis::init()
{
    LOG_FUNC();
    const size_t frame_res = fd_.get_frame_res();

    // Init CircularVideoBuffer
    vesselness_mask_env_.m0_ff_video_cb_ =
        std::make_unique<CircularVideoBuffer>(frame_res, api::get_time_window(), stream_);

    vesselness_mask_env_.m0_ff_centered_video_cb_ =
        std::make_unique<CircularVideoBuffer>(frame_res, api::get_time_window(), stream_);

    vesselness_mask_env_.vascular_pulse_video_cb_ =
        std::make_unique<CircularVideoBuffer>(frame_res, api::get_time_window(), stream_);

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
    if (err != 0)
        throw std::exception(cudaGetErrorString(cudaGetLastError()));

    shift_corners(gaussian_128_kernel_buffer_.get(), batch_size, fd_.width, fd_.height, stream_);
    cufftSafeCall(cufftExecC2C(convolution_plan_,
                               gaussian_128_kernel_buffer_.get(),
                               gaussian_128_kernel_buffer_.get(),
                               CUFFT_FORWARD));

    // (Re)Allocate vesselness buffers, can handle frame size change
    vesselness_mask_env_.time_window_ = api::get_time_window();
    vesselness_mask_env_.number_image_mean_ = 0;

    vesselness_mask_env_.m0_ff_sum_image_.resize(buffers_.gpu_postprocess_frame_size);
    vesselness_mask_env_.image_with_mean_.resize(buffers_.gpu_postprocess_frame_size);
    vesselness_mask_env_.image_centered_.resize(buffers_.gpu_postprocess_frame_size);

    vesselness_mask_env_.vascular_image_.resize(frame_res);

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

    vesselness_mask_env_.g_xx_mul_.resize(x_size * y_size);
    matrix_multiply<float>(g_xx_qy,
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

    vesselness_mask_env_.g_xy_mul_.resize(x_size * y_size);
    matrix_multiply<float>(g_xy_qy, g_xy_px, y_size, x_size, 1, vesselness_mask_env_.g_xy_mul_.get(), cublas_handler_);

    cudaXMalloc(&result_transpose, sizeof(float) * x_size * y_size);
    cublasSafeCall(cublasSgeam(cublas_handler_,
                               CUBLAS_OP_T,
                               CUBLAS_OP_N,
                               x_size,
                               y_size, // Dimensions de B
                               &alpha,
                               vesselness_mask_env_.g_xy_mul_,
                               y_size, // Source (A), pas de ligne de A
                               &beta,
                               nullptr,
                               y_size, // B est nul, donc on utilise 0
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
    vesselness_mask_env_.g_yy_mul_.resize(x_size * y_size);
    matrix_multiply<float>(g_yy_qy, g_yy_px, y_size, x_size, 1, vesselness_mask_env_.g_yy_mul_.get(), cublas_handler_);

    cudaXMalloc(&result_transpose, sizeof(float) * x_size * y_size);
    cublasSafeCall(cublasSgeam(cublas_handler_,
                               CUBLAS_OP_T,
                               CUBLAS_OP_N,
                               x_size,
                               y_size, // Dimensions de B
                               &alpha,
                               vesselness_mask_env_.g_yy_mul_,
                               y_size, // Source (A), pas de ligne de A
                               &beta,
                               nullptr,
                               y_size, // B est nul, donc on utilise 0
                               result_transpose,
                               x_size));

    vesselness_mask_env_.g_yy_mul_.reset(result_transpose);

    cudaXFree(g_yy_qy);
    cudaXFree(g_yy_px);

    // Compute Gaussian kernel for vascular pulse
    float sigma_2 = 0.02 * fd_.width;
    vesselness_mask_env_.vascular_kernel_size_ = 2 * std::ceil(2 * sigma_2) + 1;

    vesselness_mask_env_.vascular_kernel_.resize(vesselness_mask_env_.vascular_kernel_size_ *
                                                 vesselness_mask_env_.vascular_kernel_size_);
    compute_gauss_kernel(vesselness_mask_env_.vascular_kernel_, sigma_2);
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
                // Compute the flat field corrected image for each frame of the video
                convolution_kernel(buffers_.gpu_postprocess_frame,
                                   buffers_.gpu_convolution_buffer,
                                   cuComplex_buffer_.get(),
                                   &convolution_plan_,
                                   fd_.get_frame_res(),
                                   gaussian_128_kernel_buffer_.get(),
                                   true,
                                   stream_);

                vesselness_mask_env_.m0_ff_video_cb_->add_new_frame(buffers_.gpu_postprocess_frame);
                vesselness_mask_env_.m0_ff_video_cb_->compute_mean_image();

// Compute the centered image from the temporal mean of the video
#if FROM_CSV
                image_centering(vesselness_mask_env_.image_centered_,
                                buffers_.gpu_postprocess_frame,
                                m0_ff_img_csv_,
                                buffers_.gpu_postprocess_frame_size,
                                stream_);
#else
                image_centering(vesselness_mask_env_.image_centered_,
                                buffers_.gpu_postprocess_frame,
                                vesselness_mask_env_.m0_ff_video_cb_->get_mean_image(),
                                buffers_.gpu_postprocess_frame_size,
                                stream_);
#endif
                vesselness_mask_env_.m0_ff_centered_video_cb_->add_new_frame(vesselness_mask_env_.image_centered_);

            // Compute the first vesselness mask which represents both vessels types (arteries and veins)
#if FROM_CSV
                vesselness_filter(buffers_.gpu_postprocess_frame,
                                  m0_ff_img_csv_,
                                  setting<settings::VesselnessSigma>(),
                                  vesselness_mask_env_.g_xx_mul_,
                                  vesselness_mask_env_.g_xy_mul_,
                                  vesselness_mask_env_.g_yy_mul_,
                                  vesselness_mask_env_.kernel_x_size_,
                                  vesselness_mask_env_.kernel_y_size_,
                                  buffers_.gpu_postprocess_frame_size,
                                  buffers_.gpu_convolution_buffer,
                                  cuComplex_buffer_,
                                  &convolution_plan_,
                                  cublas_handler_,
                                  stream_);
#else
                vesselness_filter(buffers_.gpu_postprocess_frame,
                                  vesselness_mask_env_.m0_ff_video_cb_->get_mean_image(),
                                  setting<settings::VesselnessSigma>(),
                                  vesselness_mask_env_.g_xx_mul_,
                                  vesselness_mask_env_.g_xy_mul_,
                                  vesselness_mask_env_.g_yy_mul_,
                                  vesselness_mask_env_.kernel_x_size_,
                                  vesselness_mask_env_.kernel_y_size_,
                                  buffers_.gpu_postprocess_frame_size,
                                  buffers_.gpu_convolution_buffer,
                                  cuComplex_buffer_,
                                  &convolution_plan_,
                                  cublas_handler_,
                                  stream_);
#endif

                apply_diaphragm_mask(buffers_.gpu_postprocess_frame,
                                     fd_.width / 2 - 1,
                                     fd_.height / 2 - 1,
                                     DIAPHRAGM_FACTOR * (fd_.width + fd_.height) / 2,
                                     fd_.width,
                                     fd_.height,
                                     stream_);
            // Compute f_AVG_mean, which is the temporal average of M1 / M0
#if FROM_CSV
                compute_multiplication(vesselness_mask_env_.vascular_image_,
                                       m0_ff_img_csv_,
                                       f_avg_csv_,
                                       buffers_.gpu_postprocess_frame_size,
                                       stream_);
#endif

                // appliquer le flou
                apply_convolution(vesselness_mask_env_.vascular_image_,
                                  vesselness_mask_env_.vascular_kernel_,
                                  fd_.width,
                                  fd_.height,
                                  vesselness_mask_env_.vascular_kernel_size_,
                                  vesselness_mask_env_.vascular_kernel_size_,
                                  stream_,
                                  ConvolutionPaddingType::SCALAR,
                                  0);

                float* circle_mask;
                cudaXMalloc(&circle_mask, sizeof(float) * fd_.width * fd_.height);
                int CRV_index = compute_barycentre_circle_mask(circle_mask,
                                                               vesselness_mask_env_.vascular_image_,
                                                               buffers_.gpu_postprocess_frame_size,
                                                               stream_);

                float* bwareafilt_result;
                cudaXMalloc(&bwareafilt_result, sizeof(float) * buffers_.gpu_postprocess_frame_size);
#if FROM_CSV
                cudaXMemcpy(bwareafilt_result,
                            mask_vesselness_csv_,
                            sizeof(float) * buffers_.gpu_postprocess_frame_size,
                            cudaMemcpyDeviceToDevice);
#endif

                apply_mask_or(bwareafilt_result, circle_mask, fd_.width, fd_.height, stream_);

                bwareafilt(bwareafilt_result,
                           fd_.width,
                           fd_.height,
                           uint_buffer_1_,
                           uint_buffer_2_,
                           float_buffer_,
                           cublas_handler_,
                           stream_);

                float* mask_vesselness_clean;
                cudaXMalloc(&mask_vesselness_clean, sizeof(float) * buffers_.gpu_postprocess_frame_size);
#if FROM_CSV
                cudaXMemcpy(mask_vesselness_clean,
                            mask_vesselness_csv_,
                            sizeof(float) * buffers_.gpu_postprocess_frame_size,
                            cudaMemcpyDeviceToDevice);
#endif

                apply_mask_and(mask_vesselness_clean, bwareafilt_result, fd_.width, fd_.height, stream_);
                cudaXFree(bwareafilt_result);

                float* tmp_mult;
                cudaXMalloc(&tmp_mult, sizeof(float) * buffers_.gpu_postprocess_frame_size);

#if FROM_CSV
                compute_multiplication(tmp_mult,
                                       buffers_.gpu_postprocess_frame,
                                       mask_vesselness_clean_csv_,
                                       buffers_.gpu_postprocess_frame_size,
                                       stream_);
#endif
                vesselness_mask_env_.vascular_pulse_video_cb_->add_new_frame(tmp_mult);
                cudaXFree(tmp_mult);

                vesselness_mask_env_.vascular_pulse_video_cb_->compute_mean_video(13893);
// print_in_file_gpu(vesselness_mask_env_.vascular_pulse_video_cb_->get_mean_video(),
//                   vesselness_mask_env_.time_window_,
//                   1,
//                   "vascular_pulse",
//                   stream_);

// TODO: change hard coded values from maskvesselnessclean
// La fonction sert a rien car on importe le csv de R_VascularPulse
// Son but est de sortir R_VascularPulse, pour l'instant elle ne marche pas, faut la finir
// Elle est censé faire les etape du matlab depuis l etape "1/ 3) Compute first correlation"
#if FROM_CSV
                compute_first_correlation(
                    buffers_.gpu_postprocess_frame,
                    vesselness_mask_env_.image_centered_,
                    vascular_pulse_csv_, // vesselness_mask_env.vascular_pulse_video_cb_->get_mean_video(),
                    11727,
                    506, // vesselness_mask_env_.time_window_,
                    buffers_.gpu_postprocess_frame_size,
                    stream_);

                // // this part may be deleted as it is never used for the rest of the code
                multiply_three_vectors(vesselness_mask_env_.vascular_image_,
                                       m0_ff_img_csv_,
                                       f_avg_csv_,
                                       R_VascularPulse_csv_,
                                       buffers_.gpu_postprocess_frame_size,
                                       stream_);
#endif

                apply_convolution(vesselness_mask_env_.vascular_image_,
                                  vesselness_mask_env_.vascular_kernel_,
                                  fd_.width,
                                  fd_.height,
                                  vesselness_mask_env_.vascular_kernel_size_,
                                  vesselness_mask_env_.vascular_kernel_size_,
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

                compute_barycentre_circle_mask(circle_mask,
                                               vesselness_mask_env_.vascular_image_,
                                               buffers_.gpu_postprocess_frame_size,
                                               stream_,
                                               CRV_index);

                cudaXMalloc(&(vesselness_mask_env_.quantizedVesselCorrelation_),
                            sizeof(float) * buffers_.gpu_postprocess_frame_size);
#if FROM_CSV
                segment_vessels(vesselness_mask_env_.quantizedVesselCorrelation_,
                                R_VascularPulse_csv_,
                                mask_vesselness_clean_csv_,
                                buffers_.gpu_postprocess_frame_size,
                                stream_);
#else
            // TODO
#endif

                cudaXFree(mask_vesselness_clean);
                cudaXFree(circle_mask);
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
                cudaXFree(vesselness_mask_env_.quantizedVesselCorrelation_);
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

} // namespace holovibes::compute
