#include "laplacian_kernels.cuh"
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cmath>

// Définition du functor square.
__host__ __device__ float square::operator()(const float& x) const { return x * x; }

__global__ void laplacianKernel8(const uint8_t* input, float* output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1)
    {
        float center = static_cast<float>(input[idx]);
        float up = static_cast<float>(input[idx - width]);
        float down = static_cast<float>(input[idx + width]);
        float left = static_cast<float>(input[idx - 1]);
        float right = static_cast<float>(input[idx + 1]);
        output[idx] = up + down + left + right - 4.0f * center;
    }
    else if (x < width && y < height)
    {
        output[idx] = 0.0f;
    }
}

__global__ void laplacianKernel16(const uint16_t* input, float* output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1)
    {
        float center = static_cast<float>(input[idx]);
        float up = static_cast<float>(input[idx - width]);
        float down = static_cast<float>(input[idx + width]);
        float left = static_cast<float>(input[idx - 1]);
        float right = static_cast<float>(input[idx + 1]);
        output[idx] = up + down + left + right - 4.0f * center;
    }
    else if (x < width && y < height)
    {
        output[idx] = 0.0f;
    }
}

float processFrameCUDA(const void* frameData, int width, int height, int depth)
{
    int numPixels = width * height;

    // Allocation de la mémoire GPU pour l'image d'entrée et pour les résultats.
    void* d_input = nullptr;
    float* d_laplacian = nullptr;
    size_t inputSize = numPixels * depth;
    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_laplacian, numPixels * sizeof(float));

    // Copie de l'image depuis la mémoire hôte vers le GPU.
    cudaMemcpy(d_input, frameData, inputSize, cudaMemcpyHostToDevice);

    // Définition de la grille et des blocs.
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Lancement du kernel en fonction de la profondeur (en octets par pixel).
    if (depth == 1) // Par exemple, 8 bits.
    {
        laplacianKernel8<<<gridSize, blockSize>>>(static_cast<const uint8_t*>(d_input), d_laplacian, width, height);
    }
    else if (depth == 2) // Par exemple, 16 bits.
    {
        laplacianKernel16<<<gridSize, blockSize>>>(static_cast<const uint16_t*>(d_input), d_laplacian, width, height);
    }
    else
    {
        cudaFree(d_input);
        cudaFree(d_laplacian);
        return 0.0f;
    }
    cudaDeviceSynchronize();

    // Réduction du tableau de résultats avec Thrust pour obtenir la variance.
    thrust::device_ptr<float> dev_ptr(d_laplacian);
    float sum = thrust::reduce(dev_ptr, dev_ptr + numPixels, 0.0f, thrust::plus<float>());
    float sumSquares = thrust::transform_reduce(dev_ptr, dev_ptr + numPixels, square(), 0.0f, thrust::plus<float>());

    float mean = sum / numPixels;
    float variance = (sumSquares / numPixels) - (mean * mean);

    cudaFree(d_input);
    cudaFree(d_laplacian);

    return variance;
}