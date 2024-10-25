#include "reduce.cuh"
#include "map.cuh"
#include "tools_compute.cuh"

#include <stdio.h>

#define AUTO_CONTRAST_COMPENSATOR 10000

__global__ void
kernel_complex_divide(cuComplex* image, const uint frame_res, const float divider, const uint batch_size)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < frame_res)
    {
        for (uint i = 0; i < batch_size; ++i)
        {
            const uint batch_index = index + i * frame_res;

            image[batch_index].x /= divider;
            image[batch_index].y /= divider;
        }
    }
}

__global__ void
kernel_multiply_frames_complex(cuComplex* output, const cuComplex* input1, const cuComplex* input2, const uint size)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
    {
        const float new_x = (input1[index].x * input2[index].x) - (input1[index].y * input2[index].y);
        const float new_y = (input1[index].y * input2[index].x) + (input1[index].x * input2[index].y);
        output[index].x = new_x;
        output[index].y = new_y;
    }
}

__global__ void
kernel_divide_frames_float(const float* numerator, const float* denominator, float* output, const uint size)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
    {
        const float new_x = numerator[index] / denominator[index];
        output[index] = new_x;
    }
}

__global__ void kernel_tensor_multiply_vector(float* output,
                                              const float* tensor,
                                              const float* vector,
                                              const size_t frame_res,
                                              const ushort f_start,
                                              const ushort f_end)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= frame_res)
        return;

    float val = 0.0f;
    for (uint i = f_start; i <= f_end; i++)
    {
        const float* current_frame = tensor + i * frame_res;
        val += current_frame[index] * vector[i];
    }

    output[index] = val;
}

void multiply_frames_complex(
    const cuComplex* input1, const cuComplex* input2, cuComplex* output, const uint size, const cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    kernel_multiply_frames_complex<<<blocks, threads, 0, stream>>>(output, input1, input2, size);
    cudaCheckError();
}

void gpu_normalize(float* const input,
                   double* const result_reduce,
                   const size_t frame_res,
                   const uint norm_constant,
                   const cudaStream_t stream)
{
    reduce_add(input, result_reduce, frame_res, stream);

    /* Let x be a pixel, after renormalization
    ** x = x * 2^(norm_constant) / mean
    ** x = x * 2^(norm_constant) * frame_res / reduce_result
    ** x = x * 2^(norm_constant) * (frame_res / reduce_result)
    */
    const float multiplier = (1 << norm_constant);
    auto map_function = [multiplier, frame_res, result_reduce] __device__(const float input_pixel) -> float
    {
        /* Computing on double is really slow on a GPU, in our case
         *result_reduce can never overflow
         ** Thus it can be casted to a float
         */
        return input_pixel * multiplier * (frame_res / static_cast<const float>(*result_reduce));
    };

    map_generic(input, input, frame_res, map_function, stream);
}

void tensor_multiply_vector(float* output,
                            const float* tensor,
                            const float* vector,
                            const size_t frame_res,
                            const ushort f_start,
                            const ushort f_end,
                            const cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(frame_res, threads);
    kernel_tensor_multiply_vector<<<blocks, threads, 0, stream>>>(output, tensor, vector, frame_res, f_start, f_end);
    cudaCheckError();
}

// __global__ void kernel_translation(float* input, float* output, uint width, uint height, int shift_x, int shift_y)
// {
//     const uint index = blockIdx.x * blockDim.x + threadIdx.x;
//     if (index < width * height)
//     {
//         const int new_x = index % width;
//         const int new_y = index / width;
//         const int old_x = (new_x - shift_x + width) % width;
//         const int old_y = (new_y - shift_y + height) % height;
//         output[index] = input[old_y * width + old_x];
//     }
// }

__global__ void circshift_kernel(const float* src, float* dst, int width, int height, int shift_x, int shift_y)
{
    // Calcul des coordonnées globales du thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        // Calcul des nouvelles coordonnées en appliquant le décalage
        int new_x = (x + shift_x) % width;
        int new_y = (y + shift_y) % height;

        // Ajustement pour des décalages négatifs
        if (new_x < 0)
            new_x += width;
        if (new_y < 0)
            new_y += height;

        // Copie de l'élément source vers la nouvelle position dans la destination
        dst[new_y * width + new_x] = src[y * width + x];
    }
}

void complex_translation(float* frame, uint width, uint height, int shift_x, int shift_y, cudaStream_t stream)
{
    // We have to use a temporary buffer to avoid overwriting pixels that haven't moved yet
    float* tmp_buffer;
    if (cudaMalloc(&tmp_buffer, width * height * sizeof(float)) != cudaSuccess)
    {
        LOG_ERROR("Can't callocate buffer for repositioning");
        return;
    }
    // const uint threads = get_max_threads_1d();
    // const uint blocks = map_blocks_to_problem(width * height, threads);
    uint threads_2d = get_max_threads_2d();
    dim3 lthreads(threads_2d, threads_2d);
    dim3 lblocks(1 + (width - 1) / threads_2d, 1 + (height - 1) / threads_2d);

    // kernel_translation<<<blocks, threads, 0, stream>>>(frame, tmp_buffer, width, height, shift_x, shift_y);
    circshift_kernel<<<lblocks, lthreads, 0, stream>>>(frame, tmp_buffer, width, height, shift_x, shift_y);
    cudaCheckError();
    cudaStreamSynchronize(stream);
    cudaMemcpy(frame, tmp_buffer, width * height * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaFree(tmp_buffer);
}

__global__ void find_max(float* d_data, int size, float* max_val, int* max_idx)
{
    __shared__ float shared_max[256];
    __shared__ int shared_idx[256];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int local_tid = threadIdx.x;

    // Initialiser le maximum local et l'index
    float local_max = -FLT_MAX;
    int local_idx = -1;

    // Recherche du maximum local
    if (tid < size)
    {
        local_max = d_data[tid];
        local_idx = tid;
    }

    shared_max[local_tid] = local_max;
    shared_idx[local_tid] = local_idx;

    __syncthreads();

    // Réduction pour trouver le maximum
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (local_tid < stride)
        {
            if (shared_max[local_tid] < shared_max[local_tid + stride])
            {
                shared_max[local_tid] = shared_max[local_tid + stride];
                shared_idx[local_tid] = shared_idx[local_tid + stride];
            }
        }
        __syncthreads();
    }

    // Écriture du maximum global
    if (local_tid == 0)
    {
        max_val[blockIdx.x] = shared_max[0];
        max_idx[blockIdx.x] = shared_idx[0];
    }
}

// void compute_max(float* d_data, int size, cudaStream_t stream, float* max_val, int* max_idx) {}
void compute_max(float* d_data, int size, cudaStream_t stream, float* max_val, int* max_idx)
{
    int blockSize = 256; // Taille du bloc
    int numBlocks = (size + blockSize - 1) / blockSize;

    // Allocation de la mémoire pour les résultats intermédiaires
    float* d_max_val;
    int* d_max_idx;

    cudaMalloc(&d_max_val, numBlocks * sizeof(float));
    cudaMalloc(&d_max_idx, numBlocks * sizeof(int));

    // Lancer le noyau
    find_max<<<numBlocks, blockSize, 0, stream>>>(d_data, size, d_max_val, d_max_idx);

    // Récupérer le maximum global
    float h_max_val = -FLT_MAX;
    int h_max_idx = -1;

    float* h_results_val = new float[numBlocks];
    int* h_results_idx = new int[numBlocks];

    cudaMemcpy(h_results_val, d_max_val, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_results_idx, d_max_idx, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);

    // Trouver le maximum global
    for (int i = 0; i < numBlocks; ++i)
    {
        if (h_results_val[i] > h_max_val)
        {
            h_max_val = h_results_val[i];
            h_max_idx = h_results_idx[i];
        }
    }

    // Copie des résultats finaux
    *max_val = h_max_val;
    *max_idx = h_max_idx;

    // Nettoyage
    delete[] h_results_val;
    delete[] h_results_idx;
    cudaFree(d_max_val);
    cudaFree(d_max_idx);
}