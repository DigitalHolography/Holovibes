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

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>

using holovibes::cuda_tools::CufftHandle;

#define DIAPHRAGM_FACTOR 0.4f

namespace holovibes::compute
{
#pragma region tools
namespace
{
std::vector<float> load_gaussian_128_convolution_matrix()
{
    // There is no file None.txt for convolution
    std::vector<float> convo_matrix = {};
    const std::string& file = "gaussian_128_128_1.txt";
    auto& holo = holovibes::Holovibes::instance();

    try
    {
        auto path_file = GET_EXE_DIR / __CONVOLUTION_KERNEL_FOLDER_PATH__ / file; //"convolution_kernels" / file;
        std::string path = path_file.string();

        std::vector<float> matrix;
        uint matrix_width = 0;
        uint matrix_height = 0;
        uint matrix_z = 1;

        // Doing this the C way because it's faster
        FILE* c_file;
        fopen_s(&c_file, path.c_str(), "r");

        if (c_file == nullptr)
        {
            fclose(c_file);
            throw std::runtime_error("Invalid file path");
        }

        // Read kernel dimensions
        if (fscanf_s(c_file, "%u %u %u;", &matrix_width, &matrix_height, &matrix_z) != 3)
        {
            fclose(c_file);
            throw std::runtime_error("Invalid kernel dimensions");
        }

        size_t matrix_size = matrix_width * matrix_height * matrix_z;
        matrix.resize(matrix_size);

        // Read kernel values
        for (size_t i = 0; i < matrix_size; ++i)
        {
            if (fscanf_s(c_file, "%f", &matrix[i]) != 1)
            {
                fclose(c_file);
                throw std::runtime_error("Missing values");
            }
        }

        fclose(c_file);

        // Reshape the vector as a (nx,ny) rectangle, keeping z depth
        const uint output_width = holo.get_gpu_output_queue()->get_fd().width;
        const uint output_height = holo.get_gpu_output_queue()->get_fd().height;
        const uint size = output_width * output_height;

        // The convo matrix is centered and padded with 0 since the kernel is
        // usally smaller than the output Example: kernel size is (2, 2) and
        // output size is (4, 4) The kernel is represented by 'x' and
        //  | 0 | 0 | 0 | 0 |
        //  | 0 | x | x | 0 |
        //  | 0 | x | x | 0 |
        //  | 0 | 0 | 0 | 0 |
        const uint first_col = (output_width / 2) - (matrix_width / 2);
        const uint last_col = (output_width / 2) + (matrix_width / 2);
        const uint first_row = (output_height / 2) - (matrix_height / 2);
        const uint last_row = (output_height / 2) + (matrix_height / 2);

        convo_matrix.resize(size, 0.0f);

        uint kernel_indice = 0;
        for (uint i = first_row; i < last_row; i++)
        {
            for (uint j = first_col; j < last_col; j++)
            {
                (convo_matrix)[i * output_width + j] = matrix[kernel_indice];
                kernel_indice++;
            }
        }
    }
    catch (std::exception& e)
    {
        LOG_ERROR("Couldn't load convolution matrix : {}", e.what());
        return {};
    };
    return convo_matrix;
}

} // namespace
#pragma endregion tools

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
    std::vector<float> gaussian_kernel = load_gaussian_128_convolution_matrix();

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
    vesselness_mask_env_.m0_ff_video_.resize(buffers_.gpu_postprocess_frame_size * vesselness_mask_env_.time_window_);
    vesselness_mask_env_.image_with_mean_.resize(buffers_.gpu_postprocess_frame_size);
    vesselness_mask_env_.image_centered_.resize(buffers_.gpu_postprocess_frame_size);

    vesselness_mask_env_.vascular_image_.resize(frame_res);

    // Compute gaussian deriviatives kernels according to simga
    float sigma = api::get_vesselness_sigma();

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

    // calculer notre kernel
    float sigma_2 = 0.02 * fd_.width;
    vesselness_mask_env_.vascular_kernel_size_ = 2 * std::ceil(2 * sigma_2) + 1;

    // float* k = compute_kernel(sigma_2);

    // float* d_k;
    // cudaXMalloc(&d_k, sizeof(float) * kernel_size * kernel_size);
    // cudaXMemcpy(d_k, k, sizeof(float) * kernel_size * kernel_size, cudaMemcpyHostToDevice);
    // delete[] k;

    vesselness_mask_env_.vascular_kernel_.resize(vesselness_mask_env_.vascular_kernel_size_ *
                                                 vesselness_mask_env_.vascular_kernel_size_);
    compute_kernel_cuda(vesselness_mask_env_.vascular_kernel_, sigma_2);

    // print_in_file_cpu(vesselness_mask_env_.vascular_kernel_, 2, 2, "vascular_kernel");

    // // Pad it with zero to equal frame size
    // float* vascular_kernel_padded;
    // cudaXMalloc(&vascular_kernel_padded, sizeof(float) * fd_.height * fd_.width);
    // convolution_kernel_add_padding(vascular_kernel_padded, vascular_kernel_float, w, h, fd_.width, fd_.height,
    // stream_);

    // cudaXFree(vascular_kernel_float);

    // // Convert from float to cuComplex for FFT
    // vesselness_mask_env_.vascular_kernel_.resize(frame_res);
    // cudaXMemsetAsync(vesselness_mask_env_.vascular_kernel_, 0, frame_res * sizeof(cuComplex), stream_);
    // cudaSafeCall(cudaMemcpy2DAsync(vesselness_mask_env_.vascular_kernel_,
    //                                sizeof(cuComplex),
    //                                vascular_kernel_padded,
    //                                sizeof(float),
    //                                sizeof(float),
    //                                frame_res,
    //                                cudaMemcpyDeviceToDevice,
    //                                stream_));
    // cudaXStreamSynchronize(stream_);
    // shift_corners(vesselness_mask_env_.vascular_kernel_, batch_size, fd_.width, fd_.height, stream_);
    // cufftSafeCall(cufftExecC2C(convolution_plan_,
    //                            vesselness_mask_env_.vascular_kernel_,
    //                            vesselness_mask_env_.vascular_kernel_,
    //                            CUFFT_FORWARD));
}

void Analysis::dispose()
{
    LOG_FUNC();

    buffers_.gpu_convolution_buffer.reset(nullptr);
    cuComplex_buffer_.reset(nullptr);
    gaussian_128_kernel_buffer_.reset(nullptr);

    uint_buffer_1_.reset(nullptr);
    uint_buffer_2_.reset(nullptr);
    float_buffer_.reset(nullptr);
}

void Analysis::insert_show_artery()
{
    LOG_FUNC();

    fn_compute_vect_.conditional_push_back([=]() {
        if (setting<settings::ImageType>() == ImgType::Moments_0 && setting<settings::ArteryMaskEnabled>())
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

            // Compute an image with the temporal mean of the video
            temporal_mean(vesselness_mask_env_.image_with_mean_,
                          buffers_.gpu_postprocess_frame,
                          &vesselness_mask_env_.number_image_mean_,
                          vesselness_mask_env_.m0_ff_video_,
                          vesselness_mask_env_.m0_ff_sum_image_,
                          vesselness_mask_env_.time_window_,
                          buffers_.gpu_postprocess_frame_size,
                          stream_);

            // Compute the centered image from the temporal mean of the video
            image_centering(vesselness_mask_env_.image_centered_,
                            vesselness_mask_env_.image_with_mean_,
                            buffers_.gpu_postprocess_frame,
                            buffers_.gpu_postprocess_frame_size,
                            stream_);

            // Compute the firsy vesselness mask with represent all veisels (arteries and veins)
            vesselness_filter(buffers_.gpu_postprocess_frame,
                              m0_ff_img_csv_, // vesselness_mask_env_.image_with_mean_,
                              api::get_vesselness_sigma(),
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

            apply_diaphragm_mask(buffers_.gpu_postprocess_frame,
                                 fd_.width / 2,
                                 fd_.height / 2,
                                 DIAPHRAGM_FACTOR * (fd_.width + fd_.height) / 2,
                                 fd_.width,
                                 fd_.height,
                                 stream_);

            // DEBUGING: print in a file the final output
            // print_in_file(buffers_.gpu_postprocess_frame,
            //               buffers_.gpu_postprocess_frame_size,
            //               "final_result",
            //               stream_);
        }
    });
}

void Analysis::insert_barycentres()
{
    LOG_FUNC();

    fn_compute_vect_.conditional_push_back([=]() {
        if (setting<settings::ImageType>() == ImgType::Moments_0 && setting<settings::VeinMaskEnabled>())
        // Compute f_AVG_mean, which is the temporal average of M1 / M0
        {

            compute_multiplication(vesselness_mask_env_.vascular_image_,
                                   m0_ff_img_csv_,
                                   f_avg_csv_,
                                   buffers_.gpu_postprocess_frame_size,
                                   stream_);

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
            compute_barycentre_circle_mask(circle_mask,
                                           vesselness_mask_env_.vascular_image_,
                                           buffers_.gpu_postprocess_frame_size,
                                           stream_);

            // TODO: change hard coded values from maskvesselnessclean
            compute_first_correlation(buffers_.gpu_postprocess_frame,
                                      vesselness_mask_env_.image_centered_,
                                      vascular_pulse_csv_,
                                      11862,
                                      506,
                                      buffers_.gpu_postprocess_frame_size,
                                      stream_);

            // cudaXMemcpy(buffers_.gpu_postprocess_frame,
            //             circle_mask,
            //             buffers_.gpu_postprocess_frame_size * sizeof(float),
            //             cudaMemcpyDeviceToDevice);
            // cudaXFree(circle_mask);
        }
    });
}

void Analysis::insert_otsu()
{
    LOG_FUNC();

    fn_compute_vect_.conditional_push_back([=]() {
        if (setting<settings::ImageType>() == ImgType::Moments_0 && setting<settings::OtsuEnabled>() == true)
        {

            cublasHandle_t& handle = cuda_tools::CublasHandle::instance();
            int maxI = -1;
            int minI = -1;
            cublasIsamax(handle, buffers_.gpu_postprocess_frame_size, buffers_.gpu_postprocess_frame, 1, &maxI);
            cublasIsamin(handle, buffers_.gpu_postprocess_frame_size, buffers_.gpu_postprocess_frame, 1, &minI);

            float h_min, h_max;
            cudaXMemcpy(&h_min, buffers_.gpu_postprocess_frame + (minI - 1), sizeof(float), cudaMemcpyDeviceToHost);
            cudaXMemcpy(&h_max, buffers_.gpu_postprocess_frame + (maxI - 1), sizeof(float), cudaMemcpyDeviceToHost);

            normalise(buffers_.gpu_postprocess_frame, h_min, h_max, buffers_.gpu_postprocess_frame_size, stream_);

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
        }
    });
}

void Analysis::insert_bwareafilt()
{
    LOG_FUNC();

    fn_compute_vect_.conditional_push_back([=]() {
        if (setting<settings::ImageType>() == ImgType::Moments_0 && GET_SETTING(BwareaopenEnabled))
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

    fn_compute_vect_.conditional_push_back([=]() {
        if (setting<settings::ImageType>() == ImgType::Moments_0 &&
            setting<settings::BwareafiltEnabled>() == true) // TODO
        {
            float* image_d = buffers_.gpu_postprocess_frame.get();
            uint* labels_d = uint_buffer_1_.get();
            uint* linked_d = uint_buffer_2_.get();
            float* labels_sizes_d = float_buffer_.get();

            uint p = 30; // TODO

            get_connected_component(labels_d, labels_sizes_d, linked_d, image_d, fd_.width, fd_.height, stream_);
            area_open(image_d, labels_d, labels_sizes_d, buffers_.gpu_postprocess_frame_size, p, stream_);
        }
    });
}

} // namespace holovibes::compute
