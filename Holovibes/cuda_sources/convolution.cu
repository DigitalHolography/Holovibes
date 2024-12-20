#pragma once
#include "convolution.cuh"
#include "fresnel_transform.cuh"
#include "tools.cuh"
#include "tools_compute.cuh"
#include "complex_utils.cuh"
#include "logger.hh"
#include "common.cuh"
#include "cuda_memory.cuh"
#include <cufft.h>
#include "cuda_tools\unique_ptr.hh"
#include "cuda_tools\array.hh"
#include "cuda_tools\cufft_handle.hh"
#include <npp.h>

#include <thrust/device_ptr.h>
#include <thrust/reverse.h>
#include <thrust/device_vector.h>
#include "matrix_operations.hh"
using holovibes::cuda_tools::CufftHandle;

void convolution_kernel(float* input_output,
                        float* gpu_convolved_buffer,
                        cuComplex* cuComplex_buffer,
                        CufftHandle* plan,
                        const size_t size,
                        const cuComplex* gpu_kernel,
                        const bool divide_convolution_enabled,
                        const cudaStream_t stream)
{
    const uint threads = get_max_threads_1d();
    const uint blocks = map_blocks_to_problem(size, threads);

    /* Copy input_output (float*) to cuComplex_buffer (cuComplex*)
     * We only want to copy the float value as real part float number in the
     * cuComplex_buffer To skip the imaginary part, we use a pitch (skipped
     * data) of size sizeof(float)
     *
     * The value are first all set to 0 (real & imaginary)
     * Then value are copied 1 by 1 from input_output into the real part
     * Imaginary is skipped and thus left to its value
     */
    cudaXMemsetAsync(cuComplex_buffer, 0, size * sizeof(cuComplex), stream);
    cudaSafeCall(cudaMemcpy2DAsync(cuComplex_buffer,  // Destination memory address
                                   sizeof(cuComplex), // Pitch of destination memory
                                   input_output,      // Source memory address
                                   sizeof(float),     // Pitch of source memory
                                   sizeof(float),     // Width of matrix transfer (columns in bytes)
                                   size,              // Height of matrix transfer (rows)
                                   cudaMemcpyDeviceToDevice,
                                   stream));
    // At this point, cuComplex_buffer is the same as the input

    cufftSafeCall(cufftExecC2C(plan->get(), cuComplex_buffer, cuComplex_buffer, CUFFT_FORWARD));
    // At this point, cuComplex_buffer is the FFT of the input

    complex_hadamard_product(cuComplex_buffer, cuComplex_buffer, gpu_kernel, size, stream);
    // At this point, cuComplex_buffer is the FFT of the input multiplied by the
    // FFT of the kernel

    cufftSafeCall(cufftExecC2C(plan->get(), cuComplex_buffer, cuComplex_buffer, CUFFT_INVERSE));

    if (divide_convolution_enabled)
    {
        kernel_complex_to_modulus<<<blocks, threads, 0, stream>>>(gpu_convolved_buffer, cuComplex_buffer, size);
        cudaCheckError();
        kernel_divide_frames_float<<<blocks, threads, 0, stream>>>(input_output,
                                                                   input_output,
                                                                   gpu_convolved_buffer,
                                                                   size);
    }
    else
    {
        kernel_complex_to_modulus<<<blocks, threads, 0, stream>>>(input_output, cuComplex_buffer, size);
    }
    cudaCheckError();
}

// __global__ void flipInPlaceKernel(cufftComplex* data, int n)
// {
//     int idx = threadIdx.x + blockIdx.x * blockDim.x;

//     // Traiter uniquement la première moitié du tableau
//     if (idx < n / 2)
//     {
//         int oppositeIdx = n - 1 - idx; // Calcul de l'indice opposé

//         // Échanger les éléments
//         cufftComplex temp = data[idx];
//         data[idx] = data[oppositeIdx];
//         data[oppositeIdx] = temp;
//     }
// }

// void flip(cufftComplex* data, int n, cudaStream_t stream)
// {
//     uint threads = get_max_threads_1d();
//     uint blocks = map_blocks_to_problem(n, threads);
//     flipInPlaceKernel<<<blocks, threads, 0, stream>>>(data, n);
//     cudaCheckError();
// }

__global__ void flipInPlaceKernel(cufftComplex* data, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Traiter uniquement la première moitié du tableau
    if (idx < n / 2)
    {
        int oppositeIdx = n - 1 - idx; // Calcul de l'indice opposé

        // Échanger les éléments
        cufftComplex temp = data[idx];
        data[idx] = data[oppositeIdx];
        data[oppositeIdx] = temp;
    }
}

void flip(cufftComplex* data, int cols, int rows, cudaStream_t stream)
{
    // uint threads = get_max_threads_1d();
    // uint blocks = map_blocks_to_problem(n, threads);
    // flipInPlaceKernel<<<blocks, threads, 0, stream>>>(data, n);
    // cudaCheckError();

    thrust::device_ptr<cufftComplex> d_matrix(data);
    int size = cols * rows;
    // Inversion des données avec thrust::reverse
    // thrust::reverse(d_ptr, d_ptr + n);
    // thrust::device_pointer_cast(data + 512));

    // thrust::device_vector<float> d_matrix = h_matrix;

    // Étape 1 : Flip dans la dimension 1 (lignes)
    for (int col = 0; col < cols; ++col)
    {
        // Inverser chaque colonne
        thrust::reverse(d_matrix + col * rows, d_matrix + (col + 1) * rows);
    }

    // Étape 2 : Flip dans la dimension 2 (colonnes)
    // thrust::device_vector<cufftComplex> d_temp(size);
    // for (int row = 0; row < rows; ++row)
    // {
    //     for (int col = 0; col < cols; ++col)
    //     {
    //         // Déplacer les éléments de manière inversée
    //         d_temp[row * cols + col] = d_matrix[row * cols + (cols - col - 1)];
    //     }
    // }

    // Remplacer la matrice originale par la matrice transposée
    // d_matrix = d_temp;
    // thrust::copy(d_temp.begin(), d_temp.end(), data);
}

// void xcorr2(float* output,
//             float* input1,
//             float* input2,
//             cufftComplex* d_freq_1,
//             cufftComplex* d_freq_2,
//             cufftHandle plan_2d,
//             cufftHandle plan_2dinv,
//             const int freq_size,
//             cudaStream_t stream)
// {
//     cufftExecR2C(plan_2d, input1, d_freq_1);
//     cufftExecR2C(plan_2d, input2, d_freq_2);

//     conjugate_complex(d_freq_1, freq_size, stream);
//     flip(d_freq_1, freq_size, stream);
//     complex_hadamard_product(d_freq_1, d_freq_1, d_freq_2, freq_size, stream);

//     cufftExecC2R(plan_2dinv, d_freq_1, output);
//     // cufftExecC2R(plan_2dinv, d_freq_2, output);
//     // cufftExecC2R(plan_2dinv, d_freq_2, output);
// }

__global__ void flip_image(cufftComplex* data, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int flipped_x = width - 1 - x;
        int flipped_y = height - 1 - y;

        int idx = y * width + x;
        int flipped_idx = flipped_y * width + flipped_x;

        // Swap the values
        cufftComplex temp = data[idx];
        data[idx] = data[flipped_idx];
        data[flipped_idx] = temp;
    }
}

template <typename T>
__global__ void verticalFlipKernel(T* d_matrix, int width, int height)
{
    // Calculer l'index 2D de chaque thread
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    // Si le thread est dans les limites de la matrice
    if (x < width && y < height / 2)
    {
        // Calculer les indices linéaires des pixels à échanger
        int topIdx = y * width + x;
        int bottomIdx = (height - y - 1) * width + x;

        // Échanger les éléments
        T temp = d_matrix[topIdx];
        d_matrix[topIdx] = d_matrix[bottomIdx];
        d_matrix[bottomIdx] = temp;
    }
}

// Fonction pour appeler le kernel et gérer les paramètres
template <typename T>
void verticalFlip(T* d_matrix, int width, int height)
{
    // Définir la taille des blocs et des grilles
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    // Lancer le kernel
    verticalFlipKernel<<<gridDim, blockDim>>>(d_matrix, width, height);

    // Vérifier les erreurs CUDA
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // Synchroniser le device
    cudaDeviceSynchronize();
}

template <typename T>
__global__ void horizontalFlipKernel(T* d_matrix, int width, int height)
{
    // Calculer l'index 2D de chaque thread
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    // Si le thread est dans les limites de la matrice
    if (x < width / 2 && y < height)
    {
        // Calculer les indices linéaires des pixels à échanger
        int leftIdx = y * width + x;
        int rightIdx = y * width + (width - x - 1);

        // Échanger les éléments
        T temp = d_matrix[leftIdx];
        d_matrix[leftIdx] = d_matrix[rightIdx];
        d_matrix[rightIdx] = temp;
    }
}

template <typename T>
void horizontalFlip(T* d_matrix, int width, int height)
{
    // Définir la taille des blocs et des grilles
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    // Lancer le kernel
    horizontalFlipKernel<<<gridDim, blockDim>>>(d_matrix, width, height);

    // Vérifier les erreurs CUDA
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // Synchroniser le device
    cudaDeviceSynchronize();
}

__global__ void rot180Kernel(cufftComplex* input, cufftComplex* output, int rows, int cols)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows)
    {
        output[(rows - 1 - y) * cols + (cols - 1 - x)] = input[y * cols + x];
    }
}

// void xcorr2(float* output,
//             float* input1,
//             float* input2,
//             cufftComplex* d_freq_1,
//             cufftComplex* d_freq_2,
//             cufftHandle plan_2d,
//             cufftHandle plan_2dinv,
//             const int freq_size,
//             int width,
//             int height,
//             cudaStream_t stream)
// {
//     cufftComplex* out_rot;
//     cudaXMalloc(&out_rot, width * height * sizeof(float));

//     dim3 blockSize(16, 16);
//     dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

//     cufftExecR2C(plan_2d, input1, d_freq_1);
//     cufftExecR2C(plan_2d, input2, d_freq_2);

//     conjugate_complex(d_freq_1, width * height, stream);
//     // rot180Kernel<<<gridSize, blockSize, 0, stream>>>(d_freq_1, out_rot, width, height);
//     cudaXStreamSynchronize(stream);
//     // horizontalFlip<float>(input2, width, height);
//     // verticalFlip<float>(input2, width, height);
//     // complex_hadamard_product(d_freq_1, d_freq_1, d_freq_2, width * height, stream);

//     cufftExecC2R(plan_2dinv, d_freq_1, output);
//     // flip(d_freq_1, width, height, stream);

//     // cufftExecC2R(plan_2dinv, d_freq_1, output);
//     // cufftExecC2R(plan_2dinv, d_freq_2, output);
//     cudaFree(out_rot);
// }

__global__ void rotate180(cufftComplex* input, cufftComplex* output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int idx_in = y * width + x;
        int idx_out = (height - y - 1) * width + (width - x - 1);
        output[idx_out] = input[idx_in];
    }
}

void xcorr2(float* output,
            float* input1,
            float* input2,
            cufftComplex* d_freq_1,
            cufftComplex* d_freq_2,
            cufftHandle plan_2d,
            cufftHandle plan_2dinv,
            const int freq_size,
            int width,
            int height,
            cudaStream_t stream)
{

    cufftComplex* d_temp;
    cudaMalloc((void**)&d_temp, width * height * sizeof(cufftComplex));

    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    cudaXStreamSynchronize(stream);

    cufftExecR2C(plan_2d, input1, d_freq_1);
    cufftExecR2C(plan_2d, input2, d_freq_2);

    conjugate_complex(d_freq_1, width * height, stream);
    cudaXStreamSynchronize(stream);

    rotate180<<<gridDim, blockDim, 0, stream>>>(d_freq_1, d_temp, width, height);
    cudaXStreamSynchronize(stream);

    cudaMemcpyAsync(d_freq_1, d_temp, width * height * sizeof(cufftComplex), cudaMemcpyDeviceToDevice, stream);
    cudaXStreamSynchronize(stream);

    cufftExecC2R(plan_2dinv, d_freq_1, output);

    cudaFree(d_temp);
}