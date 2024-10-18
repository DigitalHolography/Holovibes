#include "stabilization.cuh"

namespace
{
/*! \brief The CUDA Kernel applying the circular mask to the image. Using euclidian distance and circle formula
 *   (x^2 +  y^2 = r^2)
 *  \param[out] output The output image after mask application.
 *  \param[in] input The input image on which the mask is applied.
 *  \param[in out] pixels_mean Pointer to store the mean of the pixels inside the circle. Just process the sum in this
 *  function.
 *  \param [in out] pixels_number Pointer to store the number of pixels inside the circle.
 *  \param[in] width The width of the image.
 *  \param[in] height The height of the image.
 *  \param[in] centerX The x composite of the center of the image.
 *  \param[in] centerY The y composite of the center of the image.
 *  \param[in] radius The radius of the circle.
 */
__global__ void applyCircularMaskKernel(float* output,
                                        float* input,
                                        float* pixels_mean,
                                        uint* pixels_number,
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
            // *(pixels_mean) += input[idx];
            // atomicAdd(pixels_mean, input[idx]);
            output[idx] = input[idx];
            // *(pixels_number)++;
            // atomicAdd(pixels_number, 1);
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

    uint pixels_number = 0;
    // size_t shared_memory_size = 2 * threads_2d * threads_2d * sizeof(float);

    applyCircularMaskKernel<<<lblocks, lthreads, 0, stream>>>(output,
                                                              input,
                                                              pixels_mean,
                                                              &pixels_number,
                                                              width,
                                                              height,
                                                              centerX,
                                                              centerY,
                                                              radius);

    cudaXStreamSynchronize(stream);

    // *(pixels_mean) = *(pixels_mean) / pixels_number;
    cudaCheckError();
}