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

using holovibes::cuda_tools::CufftHandle;

#include <iostream>
#include <fstream>

void writeFloatArrayToFile(const float* array, int size, const std::string& filename)
{
    // Ouvre un fichier en mode écriture
    std::ofstream outFile(filename);

    // Vérifie que le fichier est bien ouvert
    if (!outFile)
    {
        std::cerr << "Erreur : Impossible d'ouvrir le fichier " << filename << std::endl;
        return;
    }

    // Écrit chaque élément du tableau dans le fichier, un par ligne
    for (int i = 0; i < size; ++i)
    {
        outFile << array[i] << std::endl;
    }

    // Ferme le fichier
    outFile.close();
    std::cout << "Tableau écrit dans le fichier " << filename << std::endl;
}

void write1DFloatArrayToFile(const float* array, int rows, int cols, const std::string& filename)
{
    // Open the file in write mode
    std::ofstream outFile(filename);

    // Check if the file was opened successfully
    if (!outFile)
    {
        std::cerr << "Error: Unable to open the file " << filename << std::endl;
        return;
    }

    // Write the 1D array in row-major order to the file
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            outFile << array[i * cols + j]; // Calculate index in row-major order
            if (j < cols - 1)
            {
                outFile << " "; // Separate values in a row by a space
            }
        }
        outFile << std::endl; // New line after each row
    }

    // Close the file
    outFile.close();
    std::cout << "1D array written to the file " << filename << std::endl;
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

    cuda_tools::CudaUniquePtr<float> g_xx_mul_;
    cuda_tools::CudaUniquePtr<float> g_xy_mul_;
    cuda_tools::CudaUniquePtr<float> g_yy_mul_;

    g_xx_mul_.resize(x_size * y_size);
    matrix_multiply<float>(g_xx_qy, g_xx_px, y_size, x_size, 1, g_xx_mul_.get(), cublas_handler_, CUBLAS_OP_N);

    cudaXFree(g_xx_qy);
    cudaXFree(g_xx_px);

    float* g_xy_px;
    cudaXMalloc(&g_xy_px, x_size * sizeof(float));
    comp_dgaussian(g_xy_px, x, x_size, sigma, 1, stream_);

    float* g_xy_qy;
    cudaXMalloc(&g_xy_qy, y_size * sizeof(float));
    comp_dgaussian(g_xy_qy, y, y_size, sigma, 1, stream_);

    g_xy_mul_.resize(x_size * y_size);
    matrix_multiply<float>(g_xy_qy, g_xy_px, y_size, x_size, 1, g_xy_mul_.get(), cublas_handler_);

    cudaXFree(g_xy_qy);
    cudaXFree(g_xy_px);

    float* g_yy_px;
    cudaXMalloc(&g_yy_px, x_size * sizeof(float));
    comp_dgaussian(g_yy_px, x, x_size, sigma, 0, stream_);

    float* g_yy_qy;
    cudaXMalloc(&g_yy_qy, y_size * sizeof(float));
    comp_dgaussian(g_yy_qy, x, y_size, sigma, 2, stream_);

    // Compute qy * px matrices to simply two 1D convolutions to one 2D convolution
    g_yy_mul_.resize(x_size * y_size);
    matrix_multiply<float>(g_yy_qy, g_yy_px, y_size, x_size, 1, g_yy_mul_.get(), cublas_handler_);

    cudaXFree(g_yy_qy);
    cudaXFree(g_yy_px);

    float* g_xx_mul_with_pading;
    cudaXMalloc(&g_xx_mul_with_pading, fd_.get_frame_res() * sizeof(float));
    cudaXMemset(g_xx_mul_with_pading, 0, fd_.get_frame_res() * sizeof(float));
    convolution_kernel_add_padding(g_xx_mul_with_pading,
                                   g_xx_mul_.get(),
                                   x_size,
                                   y_size,
                                   fd_.width,
                                   fd_.height,
                                   stream_);

    float* g_xy_mul_with_pading;
    cudaXMalloc(&g_xy_mul_with_pading, fd_.get_frame_res() * sizeof(float));
    cudaXMemset(g_xy_mul_with_pading, 0, fd_.get_frame_res() * sizeof(float));
    convolution_kernel_add_padding(g_xy_mul_with_pading,
                                   g_xy_mul_.get(),
                                   x_size,
                                   y_size,
                                   fd_.width,
                                   fd_.height,
                                   stream_);
    float* g_yy_mul_with_pading;
    cudaXMalloc(&g_yy_mul_with_pading, fd_.get_frame_res() * sizeof(float));
    cudaXMemset(g_yy_mul_with_pading, 0, fd_.get_frame_res() * sizeof(float));
    convolution_kernel_add_padding(g_yy_mul_with_pading,
                                   g_yy_mul_.get(),
                                   x_size,
                                   y_size,
                                   fd_.width,
                                   fd_.height,
                                   stream_);
    cudaXStreamSynchronize(stream_);

    g_xx_mul_.reset(g_xx_mul_with_pading);
    g_xy_mul_.reset(g_xy_mul_with_pading);
    g_yy_mul_.reset(g_yy_mul_with_pading);

    // Convert float kernels to cuComplex kernels
    vesselness_mask_env_.g_xx_mul_comp_.resize(fd_.get_frame_res());
    cudaXMemsetAsync(vesselness_mask_env_.g_xx_mul_comp_.get(), 0, frame_res * sizeof(cuComplex), stream_);
    cudaSafeCall(cudaMemcpy2DAsync(vesselness_mask_env_.g_xx_mul_comp_.get(),
                                   sizeof(cuComplex),
                                   g_xx_mul_with_pading,
                                   sizeof(float),
                                   sizeof(float),
                                   frame_res,
                                   cudaMemcpyDeviceToDevice,
                                   stream_));

    shift_corners(vesselness_mask_env_.g_xx_mul_comp_.get(), batch_size, fd_.width, fd_.height, stream_);
    cufftSafeCall(cufftExecC2C(convolution_plan_,
                               vesselness_mask_env_.g_xx_mul_comp_.get(),
                               vesselness_mask_env_.g_xx_mul_comp_.get(),
                               CUFFT_FORWARD));

    vesselness_mask_env_.g_xy_mul_comp_.resize(fd_.get_frame_res());
    cudaXMemsetAsync(vesselness_mask_env_.g_xy_mul_comp_.get(), 0, frame_res * sizeof(cuComplex), stream_);
    cudaSafeCall(cudaMemcpy2DAsync(vesselness_mask_env_.g_xy_mul_comp_.get(),
                                   sizeof(cuComplex),
                                   g_xy_mul_with_pading,
                                   sizeof(float),
                                   sizeof(float),
                                   frame_res,
                                   cudaMemcpyDeviceToDevice,
                                   stream_));

    vesselness_mask_env_.g_yy_mul_comp_.resize(fd_.get_frame_res());
    cudaXMemsetAsync(vesselness_mask_env_.g_yy_mul_comp_.get(), 0, frame_res * sizeof(cuComplex), stream_);
    cudaSafeCall(cudaMemcpy2DAsync(vesselness_mask_env_.g_yy_mul_comp_.get(),
                                   sizeof(cuComplex),
                                   g_yy_mul_with_pading,
                                   sizeof(float),
                                   sizeof(float),
                                   frame_res,
                                   cudaMemcpyDeviceToDevice,
                                   stream_));
    cudaXStreamSynchronize(stream_);
}

void Analysis::dispose()
{
    LOG_FUNC();

    buffers_.gpu_convolution_buffer.reset(nullptr);
    cuComplex_buffer_.reset(nullptr);
    gaussian_kernel_buffer_.reset(nullptr);
}

void Analysis::insert_show_artery()
{
    LOG_FUNC();

    fn_compute_vect_.conditional_push_back(
        [=]()
        {
            if (setting<settings::ImageType>() == ImgType::Moments_0 && setting<settings::ArteryMaskEnabled>() == true)
            {
                convolution_kernel(buffers_.gpu_postprocess_frame,
                                   buffers_.gpu_convolution_buffer,
                                   cuComplex_buffer_.get(),
                                   &convolution_plan_,
                                   fd_.get_frame_res(),
                                   gaussian_kernel_buffer_.get(),
                                   true,
                                   stream_);

                temporal_mean(vesselness_mask_env_.image_with_mean_,
                              buffers_.gpu_postprocess_frame,
                              &vesselness_mask_env_.number_image_mean_,
                              vesselness_mask_env_.buffer_m0_ff_img_,
                              vesselness_mask_env_.m0_ff_sum_image_,
                              vesselness_mask_env_.time_window_,
                              buffers_.gpu_postprocess_frame_size,
                              stream_);

                image_centering(vesselness_mask_env_.image_centered_,
                                vesselness_mask_env_.image_with_mean_,
                                buffers_.gpu_postprocess_frame,
                                buffers_.gpu_postprocess_frame_size,
                                stream_);

                vesselness_filter(buffers_.gpu_postprocess_frame,
                                  vesselness_mask_env_.image_with_mean_,
                                  api::get_vesselness_sigma(),
                                  vesselness_mask_env_.g_xx_mul_comp_,
                                  vesselness_mask_env_.g_xy_mul_comp_,
                                  vesselness_mask_env_.g_yy_mul_comp_,
                                  buffers_.gpu_postprocess_frame_size,
                                  buffers_.gpu_convolution_buffer,
                                  cuComplex_buffer_,
                                  &convolution_plan_,
                                  cublas_handler_,
                                  stream_);
                float* test_filter = new float[buffers_.gpu_postprocess_frame_size];
                cudaXMemcpyAsync(test_filter,
                                 buffers_.gpu_postprocess_frame,
                                 buffers_.gpu_postprocess_frame_size * sizeof(float),
                                 cudaMemcpyDeviceToHost,
                                 stream_);
                write1DFloatArrayToFile(test_filter,
                                        sqrt(buffers_.gpu_postprocess_frame_size),
                                        sqrt(buffers_.gpu_postprocess_frame_size),
                                        "test_filter_final_result.txt");
            }
        });
}
} // namespace holovibes::compute