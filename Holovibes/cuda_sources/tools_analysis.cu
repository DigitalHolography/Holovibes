
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
#include "matrix_operations.hh"
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <fstream>


float* load_CSV_to_float_array(const std::filesystem::path& path)
{
    std::string filename = path.string();
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
                outFile << " "; // Separate values in a row by a space
        }
        outFile << std::endl; // New line after each row
    }

    // Close the file
    outFile.close();
    std::cout << "1D array written to the file " << filename << std::endl;
}

__global__ void
kernel_padding(float* output, float* input, int height, int width, int new_width, int start_x, int start_y)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int y = idx / width;
    int x = idx % width;

    if (y < height && x < width)
        output[(start_y + y) * new_width + (start_x + x)] = input[y * width + x];
}

void convolution_kernel_add_padding(float* output,
                                    float* kernel,
                                    const int width,
                                    const int height,
                                    const int new_width,
                                    const int new_height,
                                    cudaStream_t stream)
{
    int start_x = (new_width - width) / 2;
    int start_y = (new_height - height) / 2;

    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(width * height, threads);
    kernel_padding<<<blocks, threads, 0, stream>>>(output, kernel, height, width, new_width, start_x, start_y);
}

void print_in_file_gpu(float* input, uint rows, uint col, std::string filename, cudaStream_t stream)
{
    if (input == nullptr)
        return;
    float* result = new float[rows * col];
    cudaXMemcpyAsync(result, input, rows * col * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaXStreamSynchronize(stream);
    write_1D_float_array_to_file(result, rows, col, "test_" + filename + ".txt");
}

void print_in_file_cpu(float* input, uint rows, uint col, std::string filename)
{
    if (input == nullptr)
        return;
    write_1D_float_array_to_file(input, rows, col, "test_" + filename + ".txt");
}

__global__ void kernel_normalized_list(float* output, int lim, int size)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
        output[index] = (int)index - lim;
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
        output[index] = device_comp_dgaussian(input[index], sigma, n);
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
        input_output[index] *= scalar;
}

template <typename T>
void multiply_array_by_scalar_caller(T* input_output, size_t size, T scalar, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    kernel_multiply_array_by_scalar<<<blocks, threads, 0, stream>>>(input_output, size, scalar);
}
} // namespace

void multiply_array_by_scalar(float* input_output, size_t size, float scalar, cudaStream_t stream)
{
    multiply_array_by_scalar_caller<float>(input_output, size, scalar, stream);
}

// CUDA kernel to prepare H hessian matrices
__global__ void
kernel_prepare_hessian(float* output, const float* ixx, const float* ixy, const float* iyy, const int size)
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

void prepare_hessian(
    float* output, const float* ixx, const float* ixy, const float* iyy, const int size, cudaStream_t stream)
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

__global__ void kernel_apply_mask_and(float* output, const float* input, short width, short height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int index = y * width + x;

    if (x < width && y < height)
        output[index] *= input[index];
}

void apply_mask_and(float* output, const float* input, const short width, const short height, const cudaStream_t stream)
{
    // Setting up the parallelisation.
    uint threads_2d = get_max_threads_2d();
    dim3 lthreads(threads_2d, threads_2d);
    dim3 lblocks(1 + (width - 1) / threads_2d, 1 + (height - 1) / threads_2d);

    kernel_apply_mask_and<<<lblocks, lthreads, 0, stream>>>(output, input, width, height);

    cudaXStreamSynchronize(stream);
    cudaCheckError();
}

__global__ void kernel_apply_mask_or(float* output, const float* input, short width, short height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int index = y * width + x;

    if (x < width && y < height)
        output[index] = (input[index] != 0.f) ? 1.f : output[index];
}

void apply_mask_or(float* output, const float* input, const short width, const short height, const cudaStream_t stream)
{
    // Setting up the parallelisation.
    uint threads_2d = get_max_threads_2d();
    dim3 lthreads(threads_2d, threads_2d);
    dim3 lblocks(1 + (width - 1) / threads_2d, 1 + (height - 1) / threads_2d);

    kernel_apply_mask_or<<<lblocks, lthreads, 0, stream>>>(output, input, width, height);

    cudaXStreamSynchronize(stream);
    cudaCheckError();
}

float* compute_gauss_deriviatives_kernel(
    int kernel_width, int kernel_height, float sigma, cublasHandle_t cublas_handler_, cudaStream_t stream)
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
    holovibes::compute::matrix_multiply(kernel_y,
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

    cudaXFree(x);
    cudaXFree(y);
    cudaXFree(kernel_y);
    cudaXFree(kernel_x);

    return result_transpose;
}

float* compute_kernel(float sigma)
{
    int kernel_size = 2 * std::ceil(2 * sigma) + 1;
    float* kernel = new float[kernel_size * kernel_size];
    float half_size = (kernel_size - 1.0f) / 2.0f;
    float sum = 0.0f;

    int y = 0;
    for (float i = -half_size; i <= half_size; ++i)
    {
        int x = 0;
        for (float j = -half_size; j <= half_size; ++j)
        {
            float value = std::exp(-(i * i + j * j) / (2 * sigma * sigma));

            kernel[x * kernel_size + y] = value;

            sum += value;
            x++;
        }
        y++;
    }

    for (int i = 0; i < kernel_size * kernel_size; ++i)
    {
        kernel[i] /= sum;
    }

    return kernel;
}

__global__ void kernel_compute_gauss_kernel(float* output, int kernel_size, float sigma, float* d_sum)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= kernel_size || y >= kernel_size)
        return;

    float half_size = (kernel_size - 1.0f) / 2.0f;
    float i = y - half_size;
    float j = x - half_size;
    float value = expf(-(i * i + j * j) / (2 * sigma * sigma));

    output[y * kernel_size + x] = value;

    // Atomic add to accumulate the total sum (for normalization)
    atomicAdd(d_sum, value);
}

__global__ void kernel_normalize_array(float* input_output, int kernel_size, float* d_sum)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= kernel_size || y >= kernel_size)
        return;

    // Normalize each element by the computed sum in d_sum
    input_output[y * kernel_size + x] /= *d_sum;
}

void compute_gauss_kernel(float* output, float sigma)
{
    float* d_sum;
    float initial_sum = 0.0f;
    int kernel_size = 2 * std::ceil(2 * sigma) + 1;

    // Allocate memory for sum on the device and initialize to 0
    cudaXMalloc(&d_sum, sizeof(float));
    cudaXMemcpy(d_sum, &initial_sum, sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((kernel_size + blockSize.x - 1) / blockSize.x, (kernel_size + blockSize.y - 1) / blockSize.y);

    // Launch the kernel to compute the Gaussian values
    kernel_compute_gauss_kernel<<<gridSize, blockSize>>>(output, kernel_size, sigma, d_sum);

    // Normalize the kernel using the computed sum directly on the GPU
    kernel_normalize_array<<<gridSize, blockSize>>>(output, kernel_size, d_sum);

    // Free device memory for sum
    cudaXFree(d_sum);
}

__global__ void kernel_count_non_zero(const float* const input, int* const count, int rows, int cols) 
{
    // Shared memory for partial counts
    __shared__ int partial_sum[256];
    int thread_id = threadIdx.x;
    int index = blockIdx.x * blockDim.x + thread_id;
    partial_sum[thread_id] = 0;

    // Check bounds and compute non-zero counts
    if (index < rows * cols && input[index] != 0) 
        partial_sum[thread_id] = 1;
    __syncthreads();

    // Reduce within the block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) 
    {
        if (thread_id < stride) 
            partial_sum[thread_id] += partial_sum[thread_id + stride];
        __syncthreads();
    }

    // Add partial result to global count
    if (thread_id == 0) 
        atomicAdd(count, partial_sum[0]);
}

int count_non_zero(const float* const input, const int rows, const int cols, cudaStream_t stream) 
{
    int* device_count;
    float* device_input;
    int size = rows * cols;
    int result;

    // Allocate memory on device
    cudaXMalloc((void**)&device_input, size * sizeof(float));
    cudaXMalloc((void**)&device_count, sizeof(int));

    // Copy input matrix to device
    cudaXMemcpy(device_input, input, size * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize count to 0
    cudaXMemset(device_count, 0, sizeof(int));

    // Configure kernel
    dim3 threads_per_block(256);
    dim3 blocks_per_grid((size + 255) / 256);

    // Launch kernel
    kernel_count_non_zero<<<blocks_per_grid, threads_per_block, 0, stream>>>(device_input, device_count, rows, cols);

    // Copy result back to host
    cudaXMemcpy(&result, device_count, sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaXFree(device_input);
    cudaXFree(device_count);

    return result;
}

__global__ void
kernel_divide_frames_float_inplace(float* const input_output, const float* const denominator, const uint size)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
    {
        const float new_x = input_output[index] / denominator[index];
        input_output[index] = new_x;
    }
}

void divide_frames_inplace(float* const input_output, const float* const denominator, const uint size, cudaStream_t stream)
{
    const uint threads = get_max_threads_1d();
    const uint blocks = map_blocks_to_problem(size, threads);
    kernel_divide_frames_float_inplace<<<blocks, threads, 0, stream>>>(input_output, denominator, size);
}

namespace
{
int find_max_thrust(float* input, size_t size)
{
    thrust::device_ptr<float> dev_ptr(input);
    thrust::device_ptr<float> max_ptr = thrust::max_element(dev_ptr, dev_ptr + size);
    return max_ptr - dev_ptr;
}

int find_min_thrust(float* input, size_t size)
{
    thrust::device_ptr<float> dev_ptr(input);
    thrust::device_ptr<float> min_ptr = thrust::min_element(dev_ptr, dev_ptr + size);
    return min_ptr - dev_ptr;
}
} // namespace

// Kernel to normalize an array between a given range
__global__ void kernel_normalize_array(float* input_output, size_t size, float min_range, float max_range, float min_val, float max_val)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        // Normalize to [0, 1], then scale to [min_range, max_range]
        input_output[idx] = roundf(((input_output[idx] - min_val) / (max_val - min_val)) * (max_range - min_range) + min_range);
    }
}

// Host function to normalize a device-only array
void normalize_array(float* input_output, size_t size, float min_range, float max_range, cudaStream_t stream)
{
    // Step 1: Use Thrust to find min and max values on the device
    int min_idx = find_min_thrust(input_output, size);
    int max_idx = find_max_thrust(input_output, size);

    // Copy min and max values from device memory to host
    float min_val, max_val;
    cudaXMemcpy(&min_val, input_output + min_idx, sizeof(float), cudaMemcpyDeviceToHost);
    cudaXMemcpy(&max_val, input_output + max_idx, sizeof(float), cudaMemcpyDeviceToHost);

    // Step 2: Launch kernel to normalize
    const uint threads = get_max_threads_1d();
    const uint blocks = map_blocks_to_problem(size, threads);
    kernel_normalize_array<<<blocks, threads, 0, stream>>>(input_output, size, min_range, max_range, min_val, max_val);
}


void load_bin_video_file(const std::filesystem::path& path, float* output, cudaStream_t stream)
{
    const int width = 512;
    const int height = 512;
    const int frames = 506;
    const int total_size = width * height * frames;

    // Allouer un tableau pour stocker les données
    float* video_data = new float[total_size];

    // Ouvrir le fichier binaire en mode lecture
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        std::cerr << "Erreur : Impossible d'ouvrir le fichier video.bin" << std::endl;
        return;
    }

    // Lire les données dans le tableau
    file.read(reinterpret_cast<char*>(video_data), total_size * sizeof(float));
    if (!file)
    {
        std::cerr << "Erreur : Lecture du fichier incomplète" << std::endl;
        delete[] video_data;
        return;
    }

    file.close();

    cudaXMemcpyAsync(output, video_data, sizeof(float) * total_size, cudaMemcpyHostToDevice, stream);
    delete[] video_data;
}