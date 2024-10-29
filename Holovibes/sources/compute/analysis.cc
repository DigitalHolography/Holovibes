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

namespace // petit flex
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

static void matrixMultiply(float* matA, float* matB, float* result, int rowsA, int colsA, int colsB)
{
    // Initialize result matrix to zero
    for (int i = 0; i < rowsA; ++i)
    {
        for (int j = 0; j < colsB; ++j)
        {
            result[i * colsB + j] = 0;
        }
    }

    // Perform matrix multiplication
    for (int i = 0; i < rowsA; ++i)
    {
        for (int j = 0; j < colsB; ++j)
        {
            for (int k = 0; k < colsA; ++k)
            {
                result[i * colsB + j] += matA[i * colsA + k] * matB[k * colsB + j];
            }
        }
    }
}

void Analysis::init()
{
    if (buffer_m0_ff_img_ != nullptr)
    {
        cudaXFree(buffer_m0_ff_img_);
        cudaXFree(m0_ff_sum_image_);
        cudaXFree(image_with_mean_);
        cudaXFree(image_centered_);
    }
    LOG_FUNC();
    const size_t frame_res = fd_.get_frame_res();

    // No need for memset here since it will be completely overwritten by
    // cuComplex values
    buffers_.gpu_convolution_buffer.resize(frame_res);

    // No need for memset here since it will be memset in the actual convolution
    cuComplex_buffer_.resize(frame_res);

    std::vector<float> gaussian_kernel = load_convolution_matrix();

    gpu_kernel_buffer_.resize(frame_res);
    cudaXMemsetAsync(gpu_kernel_buffer_.get(), 0, frame_res * sizeof(cuComplex), stream_);
    cudaSafeCall(cudaMemcpy2DAsync(gpu_kernel_buffer_.get(),
                                   sizeof(cuComplex),
                                   gaussian_kernel.data(),
                                   sizeof(float),
                                   sizeof(float),
                                   frame_res,
                                   cudaMemcpyHostToDevice,
                                   stream_));

    constexpr uint batch_size = 1; // since only one frame.
    // We compute the FFT of the kernel, once, here, instead of every time the
    // convolution subprocess is called
    shift_corners(gpu_kernel_buffer_.get(), batch_size, fd_.width, fd_.height, stream_);
    cufftSafeCall(cufftExecC2C(convolution_plan_, gpu_kernel_buffer_.get(), gpu_kernel_buffer_.get(), CUFFT_FORWARD));

    number_image_mean_ = 0;
    cudaXMalloc(&m0_ff_sum_image_, buffers_.gpu_postprocess_frame_size * sizeof(float));

    time_window_ = api::get_time_window();

    cudaXMalloc(&buffer_m0_ff_img_, buffers_.gpu_postprocess_frame_size * time_window_ * sizeof(float));
    cudaXMalloc(&image_with_mean_, buffers_.gpu_postprocess_frame_size * sizeof(float));
    cudaXMalloc(&image_centered_, buffers_.gpu_postprocess_frame_size * sizeof(float));
    cudaXStreamSynchronize(stream_);

    // this part if for the initialisation for the vessel filter
    float sigma = api::get_vesselness_sigma();

    int gamma = 1;
    int A = std::pow(sigma, gamma);
    int x_lim = std::ceil(4 * sigma);
    int y_lim = std::ceil(4 * sigma);
    int x_size = x_lim * 2 + 1;
    int y_size = y_lim * 2 + 1;

    float* x = new float[(x_size)];
    for (int i = 0; i <= 2 * x_lim; ++i)
    {
        x[i] = i - x_lim;
    }
    float* y = new float[(y_size)];
    for (int i = 0; i <= 2 * y_lim; ++i)
    {
        y[i] = i - y_lim;
    }

    float* g_xx_px = comp_dgaussian(x, sigma, 2, x_size);
    float* g_xx_qy = comp_dgaussian(y, sigma, 0, y_size);

    free(g_xx_mul_);
    g_xx_mul_ = new float[x_size * y_size];
    matrixMultiply(g_xx_qy, g_xx_px, g_xx_mul_, y_size, 1, x_size);

    float* g_xy_px = comp_dgaussian(x, sigma, 1, x_size);
    float* g_xy_qy = comp_dgaussian(y, sigma, 1, y_size);

    free(g_xy_mul_);
    g_xy_mul_ = new float[x_size * y_size];
    matrixMultiply(g_xy_qy, g_xy_px, g_xy_mul_, y_size, 1, x_size);

    float* g_yy_px = comp_dgaussian(x, sigma, 0, x_size);
    float* g_yy_qy = comp_dgaussian(y, sigma, 2, y_size);

    free(g_yy_mul_);
    g_yy_mul_ = new float[x_size * y_size];
    matrixMultiply(g_yy_qy, g_yy_px, g_yy_mul_, y_size, 1, x_size);

    free(g_xx_px);
    free(g_xx_qy);
    free(g_xy_px);
    free(g_xy_qy);
    free(g_yy_px);
    free(g_yy_qy);

    float* tmp = kernel_add_padding(g_xx_mul_, x_size, y_size, fd_.height, fd_.width);
    free(g_xx_mul_);
    g_xx_mul_ = tmp;

    tmp = kernel_add_padding(g_xy_mul_, x_size, y_size, fd_.height, fd_.width);
    free(g_xy_mul_);
    g_xy_mul_ = tmp;

    tmp = kernel_add_padding(g_yy_mul_, x_size, y_size, fd_.height, fd_.width);
    free(g_yy_mul_);
    g_yy_mul_ = tmp;
}

void Analysis::dispose()
{
    LOG_FUNC();

    buffers_.gpu_convolution_buffer.reset(nullptr);
    cuComplex_buffer_.reset(nullptr);
    gpu_kernel_buffer_.reset(nullptr);
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
                                   gpu_kernel_buffer_.get(),
                                   true,
                                   stream_);
                temporal_mean(image_with_mean_,
                              buffers_.gpu_postprocess_frame,
                              &number_image_mean_,
                              buffer_m0_ff_img_,
                              m0_ff_sum_image_,
                              time_window_,
                              buffers_.gpu_postprocess_frame_size,
                              stream_);

                // float* mean_copy = new float[fd_.get_frame_res()];
                // cudaXMemcpy(mean_copy, image_with_mean_, fd_.get_frame_res() * sizeof(float),
                // cudaMemcpyDeviceToHost); write1DFloatArrayToFile(mean_copy, fd_.height, fd_.width, "mean.txt");
                // free(mean_copy);

                image_centering(image_centered_,
                                image_with_mean_,
                                buffers_.gpu_postprocess_frame,
                                buffers_.gpu_postprocess_frame_size,
                                stream_);

                float* mescouilles = vesselness_filter(image_with_mean_,
                                                       api::get_vesselness_sigma(),
                                                       g_xx_mul_,
                                                       g_xy_mul_,
                                                       g_yy_mul_,
                                                       buffers_.gpu_postprocess_frame_size,
                                                       buffers_.gpu_convolution_buffer,
                                                       cuComplex_buffer_,
                                                       &convolution_plan_,
                                                       stream_);

                cudaXMemcpyAsync(buffers_.gpu_postprocess_frame,
                                 mescouilles,
                                 buffers_.gpu_postprocess_frame_size * sizeof(float),
                                 cudaMemcpyHostToDevice,
                                 stream_);
            }
        });
}
} // namespace holovibes::compute