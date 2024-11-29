#include "vascular_pulse.cuh"

#include "cuda_memory.cuh"
#include <cuda_runtime.h>
#include "tools_analysis_debug.hh"
#include "compute_env.hh"

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

__global__ void kernel_divide_constant(float* vascular_pulse, int value, size_t size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
        vascular_pulse[index] /= value;
}

void divide_constant(float* vascular_pulse, int value, size_t size, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    kernel_divide_constant<<<blocks, threads, 0, stream>>>(vascular_pulse, value, size);
    cudaCheckError();
}

__global__ void kernel_divide(float* input_output, float* denominator_array, size_t size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
        input_output[index] /= denominator_array[index];
}

void divide(float* input_output, float* denominator_array, size_t size, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    kernel_divide<<<blocks, threads, 0, stream>>>(input_output, denominator_array, size);
    cudaCheckError();
}

__global__ void kernel_multiply_constant(float* vascular_pulse, float value, size_t size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
        vascular_pulse[index] *= value;
}

void multiply_constant(float* vascular_pulse, float value, size_t size, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    kernel_multiply_constant<<<blocks, threads, 0, stream>>>(vascular_pulse, value, size);
    cudaCheckError();
}

float compute_mean(float* vascular_pulse, size_t size)
{
    thrust::device_ptr<float> d_array(vascular_pulse);

    float sum = thrust::reduce(d_array, d_array + size, 0.0f, thrust::plus<float>());

    return sum / size;
}

__global__ void kernel_subtract_constant(float* output, float* input, float value, size_t size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
        output[index] = input[index] - value;
}

void subtract_constant(float* output, float* input, float value, size_t size, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    kernel_subtract_constant<<<blocks, threads, 0, stream>>>(output, input, value, size);
    cudaCheckError();
}

__global__ void kernel_multiply_three_vectors(float* output, float* input1, float* input2, float* input3, size_t size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
        output[index] = input1[index] * input2[index] * input3[index];
}

void multiply_three_vectors(
    float* output, float* input1, float* input2, float* input3, size_t size, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    kernel_multiply_three_vectors<<<blocks, threads, 0, stream>>>(output, input1, input2, input3, size);
    cudaCheckError();
}

// __global__ void
// kernel_computeMean(const float* M0, const float* vascularPulse, float* result, int rows, int cols, int depth)
// {
//     // Calcul des indices globaux
//     int x = blockIdx.x * blockDim.x + threadIdx.x; // index de ligne
//     int y = blockIdx.y * blockDim.y + threadIdx.y; // index de colonne

//     if (x < rows && y < cols)
//     {
//         float sum = 0.0f;

//         // Somme sur la 3ème dimension
//         for (int z = 0; z < depth; ++z)
//         {
//             int index3D = x * cols + y + z * rows * cols;
//             sum += M0[index3D] * vascularPulse[z];
//         }

//         // Stocker la moyenne dans le tableau résultat
//         result[x * cols + y] = sum / depth;
//     }
// }

__global__ void kernel_computeMean(const float* M0_ff_video_centered,
                                   const float* vascularPulse_centered,
                                   float* result,
                                   int rows,
                                   int cols,
                                   int depth)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < rows * cols)
    {
        float sum = 0.0f;
        for (int z = 0; z < depth; z++)
        {
            int idx_video = z * cols * rows + index;
            int idx_pulse = z; // puisque vascularPulse_centered est 1x1xDEPTH
            sum += M0_ff_video_centered[idx_video] * vascularPulse_centered[idx_pulse];
        }
        result[index] = sum / depth;
    }
}

void computeMean(
    const float* M0, const float* vascularPulse, float* result, int rows, int cols, int depth, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(rows * cols, threads);
    // dim3 blockSize(16, 16);
    // dim3 gridSize((rows + blockSize.x - 1) / blockSize.x, (cols + blockSize.y - 1) / blockSize.y);

    kernel_computeMean<<<blocks, threads, 0, stream>>>(M0, vascularPulse, result, rows, cols, depth);
    cudaCheckError();
}

__global__ void kernel_compute_std(const float* input, float* output, int size, int depth)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        double mean = 0.0f;
        double variance = 0.0f;

        // Compute mean along the third dimension
        for (int k = 0; k < depth; ++k)
        {
            mean += input[idx + size * k];
        }
        mean /= depth;

        // Compute variance along the third dimension
        for (int k = 0; k < depth; ++k)
        {
            double diff = input[idx + size * k] - mean;
            variance += diff * diff;
        }
        variance /= depth;

        // Store the standard deviation in the output array
        output[idx] = sqrt(variance);
    }
}

void compute_std(const float* input, float* output, int size, int depth, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    kernel_compute_std<<<blocks, threads, 0, stream>>>(input, output, size, depth);
    cudaCheckError();
}

__device__ void atomicMinFloat(float* address, float val)
{
    unsigned int* address_as_int = (unsigned int*)address;
    unsigned int old = *address_as_int, assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(fminf(__int_as_float(assumed), val)));
    } while (assumed != old);
}

__device__ void atomicMaxFloat(float* address, float val)
{
    unsigned int* address_as_int = (unsigned int*)address;
    unsigned int old = *address_as_int, assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(fmaxf(__int_as_float(assumed), val)));
    } while (assumed != old);
}

// Kernel for finding min and max in a float array
__global__ void findMinMax(const float* input, float* min, float* max, int size)
{
    __shared__ float localMin;
    __shared__ float localMax;

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (threadIdx.x == 0)
    {
        localMin = input[0];
        localMax = input[0];
    }
    __syncthreads();

    if (idx < size)
    {
        atomicMinFloat(&localMin, input[idx]);
        atomicMaxFloat(&localMax, input[idx]);
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        atomicMinFloat(min, localMin);
        atomicMaxFloat(max, localMax);
    }
}

// Kernel pour normaliser les valeurs entre 0 et 1
__global__ void normalize(float* input, float min, float max, int size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < size)
    {
        input[idx] = (input[idx] - min) / (max - min);
    }
}

// Fonction principale pour normaliser une matrice en CUDA
void mat2gray(float* d_input, int size)
{
    float *d_min, *d_max;
    float h_min = FLT_MAX, h_max = -FLT_MAX;

    cudaMalloc(&d_min, sizeof(float));
    cudaMalloc(&d_max, sizeof(float));

    cudaMemcpy(d_min, &h_min, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_max, &h_max, sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    findMinMax<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_min, d_max, size);

    cudaMemcpy(&h_min, d_min, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost);

    normalize<<<blocksPerGrid, threadsPerBlock>>>(d_input, h_min, h_max, size);

    cudaFree(d_min);
    cudaFree(d_max);
}

#define PRINT_IN_FILE false

void compute_first_correlation(float* output,
                               float* M0_ff_video_centered,
                               float* vascular_pulse,
                               int nnz_mask_vesslness_clean,
                               size_t length_video,
                               VesselnessFilterStruct& filter_struct_,
                               size_t image_size,
                               cudaStream_t stream)
{
    // input are good
    if (length_video == 506 && PRINT_IN_FILE)
    {
        print_in_file_gpu(M0_ff_video_centered, 512, 512, "M0_ff_video_centered", stream);
        print_in_file_gpu(vascular_pulse, 1, 506, "vascular_pulse", stream);
        std::cout << "nnz : " << nnz_mask_vesslness_clean << std::endl;
        // Everything is good here
    }
    divide_constant(vascular_pulse, nnz_mask_vesslness_clean, length_video, stream);
    if (length_video == 506 && PRINT_IN_FILE)
    {
        print_in_file_gpu(vascular_pulse, 1, 506, "vascular_pulse_with_division", stream);
        // this is good
    }
    float vascular_mean = compute_mean(vascular_pulse, length_video);
    if (length_video == 506 && PRINT_IN_FILE)
    {
        std::cout << "vascular mean : " << vascular_mean << std::endl;
        // this is good
    }
    subtract_constant(filter_struct_.vascular_pulse_centered, vascular_pulse, vascular_mean, length_video, stream);
    if (length_video == 506 && PRINT_IN_FILE)
    {
        print_in_file_gpu(filter_struct_.vascular_pulse_centered, 1, 506, "vascular_pulse_centered", stream);
        // this is good
    }

    computeMean(M0_ff_video_centered, filter_struct_.vascular_pulse_centered, output, 512, 512, length_video, stream);
    if (length_video == 506 && PRINT_IN_FILE)
    {
        print_in_file_gpu(output, 512, 512, "result_mean", stream);
        // close enough, as it is a mean
    }
    compute_std(M0_ff_video_centered, filter_struct_.std_M0_ff_video_centered, 512 * 512, length_video, stream);
    if (length_video == 506 && PRINT_IN_FILE)
    {
        print_in_file_gpu(filter_struct_.std_M0_ff_video_centered, 512, 512, "std_M0_ff_video_centered", stream);
        // lots of move in this std, which is strange
    }
    compute_std(filter_struct_.vascular_pulse_centered,
                filter_struct_.std_vascular_pulse_centered,
                1,
                length_video,
                stream);
    if (length_video == 506 && PRINT_IN_FILE)
    {
        print_in_file_gpu(filter_struct_.std_vascular_pulse_centered, 1, 1, "std_vascular_pulse_centered", stream);
        // same issue here
    }

    float std_vascular_pulse_centered_cpu;
    cudaXMemcpy(&std_vascular_pulse_centered_cpu,
                filter_struct_.std_vascular_pulse_centered,
                sizeof(float),
                cudaMemcpyDeviceToHost);

    multiply_constant(filter_struct_.std_M0_ff_video_centered, std_vascular_pulse_centered_cpu, 512 * 512, stream);

    if (length_video == 506 && PRINT_IN_FILE)
    {
        print_in_file_gpu(filter_struct_.std_M0_ff_video_centered, 512, 512, "result_product_std", stream);
        // same issue here
    }

    divide(output, filter_struct_.std_M0_ff_video_centered, 512 * 512, stream);

    if (length_video == 506 && PRINT_IN_FILE)
    {
        print_in_file_gpu(output, 512, 512, "R_vascular_pulse", stream);
        // same issue here
    }

    // mat2gray(output, 512 * 512);
    if (length_video == 506 && PRINT_IN_FILE)
    {
        print_in_file_gpu(output, 512, 512, "R_vascular_pulse_with_mat2gray", stream);
    }
}