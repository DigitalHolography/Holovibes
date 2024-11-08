
#include "cuda_memory.cuh"
#include "common.cuh"
#include "tools_analysis.cuh"
#include "convolution.cuh"
#include "tools_conversion.cuh"
#include "unique_ptr.hh"
#include "tools_compute.cuh"
#include "logger.hh"
#include "cuComplex.h"
#include "cufft_handle.hh"
#include "cublas_handle.hh"

// TODO: fix the matrix_operations include (wtf)
namespace
{
constexpr float alpha = 1.0f;
constexpr float beta = 0.0f;
void matrix_multiply(const float* A,
                     const float* B,
                     int A_height,
                     int B_width,
                     int A_width_B_height,
                     float* C,
                     const cublasHandle_t& handle,
                     cublasOperation_t op_A,
                     cublasOperation_t op_B)
{
    cublasSafeCall(cublasSgemm(handle,
                               op_A,
                               op_B,
                               A_height,
                               B_width,
                               A_width_B_height,
                               &alpha,
                               A,
                               A_height,
                               B,
                               A_width_B_height,
                               &beta,
                               C,
                               B_width));
}
}

float* load_CSV_to_float_array(const std::string& filename)
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


void write_1D_float_array_to_file(const float* array, int rows, int cols, const std::string& filename)
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

__global__ void kernel_padding(float* output, float* input, int height, int width, int new_width, int start_x, int start_y) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int y = idx / width;
    int x = idx % width;

    if (y < height && x < width) 
    {
        output[(start_y + y) * new_width + (start_x + x)] = input[y * width + x];
    }
}


void convolution_kernel_add_padding(float* output, float* kernel, const int width, const int height, const int new_width, const int new_height, cudaStream_t stream) 
{
    int start_x = (new_width - width) / 2;
    int start_y = (new_height - height) / 2;

    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(width * height, threads);
    kernel_padding<<<blocks, threads, 0, stream>>>(output, kernel, height, width, new_width, start_x, start_y);

}

void print_in_file(float* input, uint rows, uint col, std::string filename, cudaStream_t stream)
{
    if (input == nullptr)
    {
        return;
    }
    float* result = new float[rows * col];
    cudaXMemcpyAsync(result,
                        input,
                        rows * col * sizeof(float),
                        cudaMemcpyDeviceToHost,
                        stream);
    cudaXStreamSynchronize(stream);
    write_1D_float_array_to_file(result,
                            rows,
                            col,
                            "test_" + filename + ".txt");
}

__global__ void kernel_normalized_list(float* output, int lim, int size)
{
     const int index = blockIdx.x * blockDim.x + threadIdx.x;
     if (index < size)
     {
        output[index] = (int)index - lim;
     }
}

void normalized_list(float* output, int lim, int size, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    kernel_normalized_list<<<blocks, threads, 0, stream>>>(output, lim, size);   
}

__device__ float comp_hermite(int n, float x)
{
    if (n == 0)
        return 1.0f;
    if (n == 1)
        return 2.0f * x;
    if (n > 1)
        return (2.0f * x * comp_hermite(n - 1, x)) - (2.0f * (n - 1) * comp_hermite(n - 2, x));
    return 0.0f;
}

__device__ float comp_gaussian(float x, float sigma)
{
    return 1 / (sigma * (sqrt(2 * M_PI))) * exp((-1 * x * x) / (2 * sigma * sigma));
}

__device__ float device_comp_dgaussian(float x, float sigma, int n)
{
    float A = pow((-1 / (sigma * sqrt((float)2))), n);
    float B = comp_hermite(n, x / (sigma * sqrt((float)2)));
    float C = comp_gaussian(x, sigma);
    return A * B * C;
}

__global__ void kernel_comp_dgaussian(float* output, float* input, size_t input_size, float sigma, int n)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < input_size)
    {
        output[index] = device_comp_dgaussian(input[index], sigma, n);
    }
}


void comp_dgaussian(float* output, float* input, size_t input_size, float sigma, int n, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(input_size, threads);
    kernel_comp_dgaussian<<<blocks, threads, 0, stream>>>(output, input, input_size, sigma, n);   
}

namespace
{
template <typename T>
__global__ void kernel_multiply_array_by_scalar(T* input_output, size_t size, const T scalar)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        input_output[index] *= scalar;
    }
}

template <typename T>
void multiply_array_by_scalar_caller(T* input_output, size_t size, T scalar, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    kernel_multiply_array_by_scalar<<<blocks, threads, 0, stream>>>(input_output, size, scalar);
}
}

void multiply_array_by_scalar(float* input_output, size_t size, float scalar, cudaStream_t stream)
{
    multiply_array_by_scalar_caller<float>(input_output, size, scalar, stream);
}

// CUDA kernel to prepare H hessian matrices
__global__ void kernel_prepare_hessian(float* output, const float* ixx, const float* ixy, const float* iyy, const int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
    {
        // Prepare the 2x2 submatrix for point `index`
        output[index * 3 + 0] = ixx[index];
        output[index * 3 + 1] = ixy[index];
        output[index * 3 + 2] = iyy[index];
    }
}

void prepare_hessian(float* output, const float* ixx, const float* ixy, const float* iyy, const int size, cudaStream_t stream)
{
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    kernel_prepare_hessian<<<numBlocks, blockSize, 0, stream>>>(output, ixx, ixy, iyy, size);
}

__global__ void kernel_compute_eigen(float* H, int size, float* lambda1, float* lambda2)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
    {
        double a = H[index * 3], b = H[index * 3 + 1], d = H[index * 3 + 2];
        double trace = a + d;
        double determinant = a * d - b * b;
        double discriminant = trace * trace - 4 * determinant;
        if (discriminant >= 0)
        {
            double eig1 = (trace + std::sqrt(discriminant)) / 2;
            double eig2 = (trace - std::sqrt(discriminant)) / 2;
            if (std::abs(eig1) < std::abs(eig2))
            {
                lambda1[index] = eig1;
                lambda2[index] = eig2;
            }
            else
            {
                lambda1[index] = eig2;
                lambda2[index] = eig1;
            }
        }
    }
}

void compute_eigen_values(float* H, int size, float* lambda1, float* lambda2, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    kernel_compute_eigen<<<blocks, threads, 0, stream>>>(H, size, lambda1, lambda2);
}



__global__ void
kernel_apply_diaphragm_mask(float* output, short width, short height, float center_X, float center_Y, float radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int index = y * width + x;

    if (x < width && y < height)
    {
        float distance_squared = (x - center_X) * (x - center_X) + (y - center_Y) * (y - center_Y);
        float radius_squared = radius * radius;

        // If the point is inside the circle set the value to 1.
        if (distance_squared > radius_squared)
            output[index] = 0;
    }
}

void apply_diaphragm_mask(float* output,
                       const float center_X,
                       const float center_Y,
                       const float radius,
                       const short width,
                       const short height,
                       const cudaStream_t stream)
{
    // Setting up the parallelisation.
    uint threads_2d = get_max_threads_2d();
    dim3 lthreads(threads_2d, threads_2d);
    dim3 lblocks(1 + (width - 1) / threads_2d, 1 + (height - 1) / threads_2d);

    kernel_apply_diaphragm_mask<<<lblocks, lthreads, 0, stream>>>(output, width, height, center_X, center_Y, radius);

    cudaXStreamSynchronize(stream);
    cudaCheckError();
}

__global__ void
kernel_compute_circle_mask(float* output, short width, short height, float center_X, float center_Y, float radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int index = y * width + x;

    if (x < width && y < height)
    {
        float distance_squared = (x - center_X) * (x - center_X) + (y - center_Y) * (y - center_Y);
        float radius_squared = radius * radius;

        // If the point is inside the circle set the value to 1.
        if (distance_squared <= radius_squared)
            output[index] = 1;
        else
            output[index] = 0;
    }
}

void compute_circle_mask(float* output,
                       const float center_X,
                       const float center_Y,
                       const float radius,
                       const short width,
                       const short height,
                       const cudaStream_t stream)
{
    // Setting up the parallelisation.
    uint threads_2d = get_max_threads_2d();
    dim3 lthreads(threads_2d, threads_2d);
    dim3 lblocks(1 + (width - 1) / threads_2d, 1 + (height - 1) / threads_2d);

    kernel_compute_circle_mask<<<lblocks, lthreads, 0, stream>>>(output, width, height, center_X, center_Y, radius);

    cudaXStreamSynchronize(stream);
    cudaCheckError();
}

__global__ void
kernel_apply_mask_and(float* output, const float* input, short width, short height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int index = y * width + x;

    if (x < width && y < height)
    {
        output[y * width + x] *= input[y * width + x];
    }
}

void apply_mask_and(float* output,
                       const float* input,
                       const short width,
                       const short height,
                       const cudaStream_t stream)
{
    // Setting up the parallelisation.
    uint threads_2d = get_max_threads_2d();
    dim3 lthreads(threads_2d, threads_2d);
    dim3 lblocks(1 + (width - 1) / threads_2d, 1 + (height - 1) / threads_2d);

    kernel_apply_mask_and<<<lblocks, lthreads, 0, stream>>>(output, input, width, height);

    cudaXStreamSynchronize(stream);
    cudaCheckError();
}

__global__ void
kernel_apply_mask_or(float* output, const float* input, short width, short height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int index = y * width + x;

    if (x < width && y < height)
    {
        output[index] = (input[index] != 0.f) ? 1.f : output[index];
    }
}

void apply_mask_or(float* output,
                       const float* input,
                       const short width,
                       const short height,
                       const cudaStream_t stream)
{
    // Setting up the parallelisation.
    uint threads_2d = get_max_threads_2d();
    dim3 lthreads(threads_2d, threads_2d);
    dim3 lblocks(1 + (width - 1) / threads_2d, 1 + (height - 1) / threads_2d);

    kernel_apply_mask_or<<<lblocks, lthreads, 0, stream>>>(output, input, width, height);

    cudaXStreamSynchronize(stream);
    cudaCheckError();
}

float* compute_gaussian_kernel(int kernel_width, int kernel_height, float sigma, cublasHandle_t cublas_handler_, cudaStream_t stream)
{
    // Initialize normalized centered at 0 lists, ex for kernel_width = 3 : [-1, 0, 1]
    float* x;
    cudaXMalloc(&x, kernel_width * sizeof(float));
    normalized_list(x, (kernel_width - 1) / 2, kernel_width, stream);

    float* y;
    cudaXMalloc(&y, kernel_height * sizeof(float));
    normalized_list(y, (kernel_height - 1) / 2, kernel_height, stream);

    // Initialize X and Y deriviative gaussian kernels
    float* kernel_x;
    cudaXMalloc(&kernel_x, kernel_width * sizeof(float));
    comp_dgaussian(kernel_x, x, kernel_width, sigma, 2, stream);

    float* kernel_y;
    cudaXMalloc(&kernel_y, kernel_height * sizeof(float));
    comp_dgaussian(kernel_y, y, kernel_height, sigma, 0, stream);

    cudaXStreamSynchronize(stream);

    float* kernel_result;
    cudaXMalloc(&kernel_result, sizeof(float) * kernel_width * kernel_height);
    matrix_multiply(kernel_y,
                           kernel_x,
                           kernel_height,
                           kernel_width,
                           1,
                           kernel_result,
                           cublas_handler_,
                           CUBLAS_OP_N,
                           CUBLAS_OP_N);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    float* result_transpose;
    cudaXMalloc(&result_transpose, sizeof(float) * kernel_width * kernel_height);
    cublasSafeCall(cublasSgeam(cublas_handler_,
                               CUBLAS_OP_T,
                               CUBLAS_OP_N,
                               kernel_width,
                               kernel_height,
                               &alpha,
                               kernel_result,
                               kernel_height,
                               &beta,
                               nullptr,
                               kernel_height,
                               result_transpose,
                               kernel_width));

    cudaXStreamSynchronize(stream);
    cudaXFree(kernel_result);
    kernel_result = result_transpose;

    cudaXFree(kernel_y);
    cudaXFree(kernel_x);

    return kernel_result;
}