#include "stabilization.cuh"

namespace
{
/*! \brief The CUDA Kernel applying the circular mask to the image. Using euclidian distance and circle formula
 *   (x^2 +  y^2 = r^2)
 *  \param[out] output The output image after mask application.
 *  \param[in] input The input image on which the mask is applied.
 *  \param[in out] mean_vector Vector used to store the sum of the pixels [0] and the number of pixels [1] inside the
 *  circle.
 *  \param[in] width The width of the image.
 *  \param[in] height The height of the image.
 *  \param[in] centerX The x composite of the center of the image.
 *  \param[in] centerY The y composite of the center of the image.
 *  \param[in] radius The radius of the circle.
 */
__global__ void applyCircularMaskKernel(float* output,
                                        float* input,
                                        float* mean_vector,
                                        short width,
                                        short height,
                                        float centerX,
                                        float centerY,
                                        float radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = y * width + x;

    if (x < width && y < height)
    {
        float distanceSquared = (x - centerX) * (x - centerX) + (y - centerY) * (y - centerY);
        float radiusSquared = radius * radius;

        if (distanceSquared > radiusSquared)
            output[idx] = 0.0f;
        else
        {
            output[idx] = input[idx];
            // TODO : Optimize the compute with shared memory and reduction, warp-level primitives.
            // (See reduction functions)
            atomicAdd(&mean_vector[0], input[idx]); // Pixels sum
            atomicAdd(&mean_vector[1], 1.0f);       // Pixels count
        }
    }
}
} // namespace

void applyCircularMask(
    float* output, float* input, float* pixels_mean, short width, short height, const cudaStream_t stream)
{
    // Get the center and radius of the circle.
    float centerX = width / 2.0f;
    float centerY = height / 2.0f;
    float radius = min(width, height) / 3.0f; // 3.0f could be change to get a different size for the circle.

    // Setting up the parallelisation.
    uint threads_2d = get_max_threads_2d();
    dim3 lthreads(threads_2d, threads_2d);
    dim3 lblocks(1 + (width - 1) / threads_2d, 1 + (height - 1) / threads_2d);

    // Allocating memory on GPU to compute the sum [0] and number [1] of pixels, used for mean computation.
    float* gpu_mean_vector;
    cudaMalloc(&gpu_mean_vector, 2 * sizeof(float));
    cudaMemset(gpu_mean_vector, 0, 2 * sizeof(float));

    applyCircularMaskKernel<<<lblocks, lthreads, 0, stream>>>(output,
                                                              input,
                                                              gpu_mean_vector,
                                                              width,
                                                              height,
                                                              centerX,
                                                              centerY,
                                                              radius);

    // Make sur that the mean compute is done.
    cudaXStreamSynchronize(stream);

    // Transfering memory to the CPU.
    float cpu_mean_vector[2];
    cudaMemcpy(cpu_mean_vector, gpu_mean_vector, 2 * sizeof(float), cudaMemcpyDeviceToHost);

    // Avoid zero division
    if (cpu_mean_vector[1] > 0)
        *pixels_mean = cpu_mean_vector[0] / cpu_mean_vector[1];
    else
        *pixels_mean = 0.0f;

    // Release GPU memory
    cudaFree(gpu_mean_vector);

    cudaCheckError();
}