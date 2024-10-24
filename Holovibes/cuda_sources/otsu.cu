#include "otsu's.cuh"

#define NUM_BINS 256

// CUDA kernel to calculate the histogram
__global__ void histogramKernel(float* image, int* hist, int imgSize)
{
    if (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < imgSize)
        atomicAdd(&hist[(unsigned char)image[idx]], 1);
}

// CUDA kernel to compute Otsu's between-class variance
__global__ void otsuKernel(int* hist, int imgSize, float* betweenClassVariance)
{
    __shared__ int s_hist[NUM_BINS];
    __shared__ float s_prob[NUM_BINS];

    if (int tid = threadIdx.x; tid < NUM_BINS)
    {
        s_hist[tid] = hist[tid];
        s_prob[tid] = (float)s_hist[tid] / imgSize;
    }
    __syncthreads();

    float weightBackground = 0;
    float sumBackground = 0;
    float sumTotal = 0;

    for (int i = 0; i < NUM_BINS; i++)
    {
        sumTotal += i * s_hist[i];
    }

    float maxVariance = 0;
    int optimalThreshold = 0;

    for (int t = 0; t < NUM_BINS; t++)
    {
        weightBackground += s_prob[t];
        float weightForeground = 1.0 - weightBackground;

        if (weightBackground == 0 || weightForeground == 0)
            continue;

        sumBackground += t * s_prob[t];
        float meanBackground = sumBackground / weightBackground;
        float meanForeground = (sumTotal - sumBackground) / weightForeground;

        float varianceBetween =
            weightBackground * weightForeground * (meanBackground - meanForeground) * (meanBackground - meanForeground);
        if (varianceBetween > maxVariance)
        {
            maxVariance = varianceBetween;
            optimalThreshold = t;
        }
    }

    betweenClassVariance[0] = maxVariance;
    betweenClassVariance[1] = (float)optimalThreshold;
}

// CUDA kernel to DO What i want
__global__ void myKernel(float* image, float p, int imgSize)
{
    if (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < imgSize)
        image[idx] = (image[idx] < p) ? 0 : 255;
}

// Host function to run Otsu's algorithm using CUDA
void otsuThreshold(float* image,
                   const size_t frame_res,
                   const int batch_size,
                   const camera::PixelDepth depth,
                   const cudaStream_t stream)
{
    float* d_image;
    int* d_hist;
    float* d_betweenClassVariance;
    // float h_betweenClassVariance[2];

    // Allocate memory on the GPU
    cudaMalloc(&d_image, frame_res * sizeof(float));
    cudaMalloc(&d_hist, NUM_BINS * sizeof(int));
    cudaMalloc(&d_betweenClassVariance, 2 * sizeof(float));

    // Copy image data to GPU
    cudaMemcpy(d_image, image, frame_res * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize the histogram on the GPU
    cudaMemset(d_hist, 0, NUM_BINS * sizeof(int));

    // Define block and grid sizes
    int blockSize = 256;
    int gridSize = (frame_res + blockSize - 1) / blockSize;

    // Run histogram kernel
    histogramKernel<<<gridSize, blockSize, 0, stream>>>(d_image, d_hist, frame_res); // TODO check 0 befor stram
    cudaDeviceSynchronize();

    // Run Otsu's kernel to compute the optimal threshold
    otsuKernel<<<1, NUM_BINS, 0, stream>>>(d_hist, frame_res, d_betweenClassVariance);
    cudaDeviceSynchronize();

    // Copy the result back to host
    // cudaMemcpy(h_betweenClassVariance, d_betweenClassVariance, 2 * sizeof(float), cudaMemcpyDeviceToHost);

    // TODO
    myKernel<<<gridSize, blockSize, 0, stream>>>(d_image, d_betweenClassVariance[1], frame_res);
    cudaDeviceSynchronize();

    // Copy image data to GPU
    cudaMemcpy(image, d_image, frame_res * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_image);
    cudaFree(d_hist);
    cudaFree(d_betweenClassVariance);

    // return (int)h_betweenClassVariance[1]; // Return optimal threshold
}