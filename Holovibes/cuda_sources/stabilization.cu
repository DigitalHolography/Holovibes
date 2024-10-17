#include "stabilization.cuh"

namespace
{
/*! \brief The CUDA Kernel applying the circular mask to the image. Using euclidian distance and circle formula
 *   (x^2 +  y^2 = r^2)
 *  \param[in out] image The image being processed, changes are made in place for now but will
 *  probably change.
 *  \param[in] width The width of the image.
 *  \param[in] height The height of the image.
 *  \param[in] centerX The x composite of the center of the image.
 *  \param[in] centerY The y composite of the center of the image.
 *  \param[in] radius The radius of the circle.
 */
__global__ void
applyCircularMaskKernel(float* image, short width, short height, float centerX, float centerY, float radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = y * width + x;

    if (x < width && y < height)
    {
        float distanceSquared = (x - centerX) * (x - centerX) + (y - centerY) * (y - centerY);
        float radiusSquared = radius * radius;

        if (distanceSquared > radiusSquared)
        {
            image[idx] = 0.0f;
        }
    }
}
} // namespace

void applyCircularMask(float* in_out, short width, short height, const cudaStream_t stream)
{
    // Get the center and radius of the circle.
    float centerX = width / 2.0f;
    float centerY = height / 2.0f;
    float radius = min(width, height) / 3.0f; // 3.0f could be change to get a different size for the circle.

    // Setting up the parallelisation.
    uint threads_2d = get_max_threads_2d();
    dim3 lthreads(threads_2d, threads_2d);
    dim3 lblocks(1 + (width - 1) / threads_2d, 1 + (height - 1) / threads_2d);

    applyCircularMaskKernel<<<lblocks, lthreads, 0, stream>>>(in_out, width, height, centerX, centerY, radius);

    cudaCheckError();
}