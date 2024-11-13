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
#include "m0_treatments.cuh"
#include "vesselness_filter.cuh"
#include "tools_analysis.cuh"
#include "API.hh"
#include "otsu.cuh"
#include "cublas_handle.hh"
#include "bw_area_filter.cuh"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>

using holovibes::cuda_tools::CufftHandle;

#include <iostream>
#include <fstream>

#define DIAPHRAGM_FACTOR 0.4f

float* loadCSVtoFloatArray(const std::string& filename)
{
    std::ifstream file(filename);

    if (!file.is_open())
    {
        std::cerr << "Erreur : impossible d'ouvrir le fichier " << filename << std::endl;
        return nullptr;
    }

    std::vector<float> values;
    std::string line;

    // Lire le fichier ligne par ligne
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string value;
        // Lire chaque valeur séparée par des virgules (ou espaces, selon le fichier)
        while (std::getline(ss, value, ','))
        {
            try
            {
                values.push_back(std::stof(value)); // Convertir la valeur en float et l'ajouter au vecteur
            }
            catch (const std::invalid_argument&)
            {
                std::cerr << "Erreur de conversion de valeur : " << value << std::endl;
            }
        }
    }

    file.close();

    // Copier les valeurs dans un tableau float*
    float* dataArray = new float[values.size()];
    for (int i = 0; i < values.size(); ++i)
    {
        dataArray[i] = values[i];
    }

    return dataArray;
}

namespace holovibes::compute
{

namespace
{
std::vector<float> load_convolution_matrix()
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

void Analysis::init()
{
    LOG_FUNC();
    const size_t frame_res = fd_.get_frame_res();

    // No need for memset here since it will be completely overwritten by
    // cuComplex values
    buffers_.gpu_convolution_buffer.resize(frame_res);

    // No need for memset here since it will be memset in the actual convolution
    cuComplex_buffer_.resize(frame_res);

    std::vector<float> gaussian_kernel = load_convolution_matrix();

    // Prepare gaussian blur kernel
    gaussian_kernel_buffer_.resize(frame_res);
    cudaXMemsetAsync(gaussian_kernel_buffer_.get(), 0, frame_res * sizeof(cuComplex), stream_);
    cudaSafeCall(cudaMemcpy2DAsync(gaussian_kernel_buffer_.get(),
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
    err += !uint_gpu_.resize(1);
    if (err != 0)
        throw std::exception(cudaGetErrorString(cudaGetLastError()));

    shift_corners(gaussian_kernel_buffer_.get(), batch_size, fd_.width, fd_.height, stream_);
    cufftSafeCall(
        cufftExecC2C(convolution_plan_, gaussian_kernel_buffer_.get(), gaussian_kernel_buffer_.get(), CUFFT_FORWARD));

    // Allocate vesselness buffers
    vesselness_mask_env_.time_window_ = api::get_time_window();
    vesselness_mask_env_.number_image_mean_ = 0;

    vesselness_mask_env_.m0_ff_sum_image_.resize(buffers_.gpu_postprocess_frame_size);
    vesselness_mask_env_.buffer_m0_ff_img_.resize(buffers_.gpu_postprocess_frame_size *
                                                  vesselness_mask_env_.time_window_);
    vesselness_mask_env_.image_with_mean_.resize(buffers_.gpu_postprocess_frame_size);
    vesselness_mask_env_.image_centered_.resize(buffers_.gpu_postprocess_frame_size);

    float sigma = api::get_vesselness_sigma();

    int gamma = 1;
    int A = std::pow(sigma, gamma);
    int x_lim = std::ceil(4 * sigma);
    int y_lim = std::ceil(4 * sigma);
    int x_size = x_lim * 2 + 1;
    int y_size = y_lim * 2 + 1;

    vesselness_mask_env_.kernel_x_size_ = x_size;
    vesselness_mask_env_.kernel_y_size_ = y_size;

    // Initialize normalized lists, ex for x_size = 3 : [-1, 0, 1]
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
                               y_size, // Dimensions de B
                               &alpha,
                               vesselness_mask_env_.g_xx_mul_,
                               y_size, // Source (A), pas de ligne de A
                               &beta,
                               nullptr,
                               y_size, // B est nul, donc on utilise 0
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

    float* data_csv_cpu = loadCSVtoFloatArray("C:/Users/Rakushka/Documents/Holovibes/data_n.csv");
    data_csv_.resize(frame_res);
    cudaXMemcpy(data_csv_, data_csv_cpu, frame_res * sizeof(float), cudaMemcpyHostToDevice);

    data_csv_cpu = loadCSVtoFloatArray("C:/Users/Rakushka/Documents/Holovibes/f_AVG_mean.csv");
    data_csv_avg_.resize(frame_res);
    cudaXMemcpy(data_csv_avg_, data_csv_cpu, frame_res * sizeof(float), cudaMemcpyHostToDevice);
}

void Analysis::dispose()
{
    LOG_FUNC();

    buffers_.gpu_convolution_buffer.reset(nullptr);
    cuComplex_buffer_.reset(nullptr);
    gaussian_kernel_buffer_.reset(nullptr);

    uint_buffer_1_.reset(nullptr);
    uint_buffer_2_.reset(nullptr);
    float_buffer_.reset(nullptr);
    uint_gpu_.reset(nullptr);
}

void Analysis::insert_show_artery()
{
    LOG_FUNC();

    fn_compute_vect_.conditional_push_back(
        [=]()
        {
            if (setting<settings::ImageType>() == ImgType::Moments_0 && setting<settings::ArteryMaskEnabled>() == true)
            {
                // Compute the flat field corrected video
                convolution_kernel(buffers_.gpu_postprocess_frame,
                                   buffers_.gpu_convolution_buffer,
                                   cuComplex_buffer_.get(),
                                   &convolution_plan_,
                                   fd_.get_frame_res(),
                                   gaussian_kernel_buffer_.get(),
                                   true,
                                   stream_);

                // Compute an image with the temporal mean of the video
                temporal_mean(vesselness_mask_env_.image_with_mean_,
                              buffers_.gpu_postprocess_frame,
                              &vesselness_mask_env_.number_image_mean_,
                              vesselness_mask_env_.buffer_m0_ff_img_,
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

                // DEBUGING: useful to compare result with a file NOT FOR RESULT ON THE SCREEN
                // DEBUGING: we load a temporal mean from MatLab to make sure it's he next part which is bad
                // float* data = loadCSVtoFloatArray("C:/Users/Karachayevsk/Documents/Holovibes/data_n.csv");
                // cudaXMemcpy(vesselness_mask_env_.image_with_mean_,
                //             data,
                //             buffers_.gpu_postprocess_frame_size * sizeof(float),
                //             cudaMemcpyHostToDevice);

                // Compute the firsy vesselness mask with represent all veisels (arteries and veins)
                vesselness_filter(buffers_.gpu_postprocess_frame,
                                  data_csv_, // vesselness_mask_env_.image_with_mean_,
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

void Analysis::insert_otsu()
{
    LOG_FUNC();

    fn_compute_vect_.conditional_push_back(
        [=]()
        {
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

    fn_compute_vect_.conditional_push_back(
        [=]()
        {
            if (setting<settings::ImageType>() == ImgType::Moments_0 && setting<settings::BwareafiltEnabled>() == true)
            {
                float* image_d = buffers_.gpu_postprocess_frame.get();
                uint* labels_d = uint_buffer_1_.get();
                uint* linked_d = uint_buffer_2_.get();
                float* labels_sizes_d = float_buffer_.get();

                cublasHandle_t& handle = cuda_tools::CublasHandle::instance();

                get_connected_component(labels_d,
                                        labels_sizes_d,
                                        linked_d,
                                        uint_gpu_.get(),
                                        image_d,
                                        fd_.width,
                                        fd_.height,
                                        stream_);

                int maxI = -1;
                cublasIsamax(handle, buffers_.gpu_postprocess_frame_size, labels_sizes_d, 1, &maxI);
                if (maxI - 1 > 0)
                    area_filter(image_d, labels_d, buffers_.gpu_postprocess_frame_size, maxI - 1, stream_);
            }
        });
}

} // namespace holovibes::compute
