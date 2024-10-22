#include "stabilization.cuh"

void stabilization_get_mask(float* mask, short width, short height, const cudaStream_t stream)
{
    // Get the center and radius of the circle.
    float center_X = width / 2.0f;
    float center_Y = height / 2.0f;
    float radius = min(width, height) / 3.0f; // 3.0f could be change to get a different size for the circle.

    // Setting up the parallelisation.
    uint threads_2d = get_max_threads_2d();
    dim3 lthreads(threads_2d, threads_2d);
    dim3 lblocks(1 + (width - 1) / threads_2d, 1 + (height - 1) / threads_2d);

    // Allocating memory on GPU to compute the sum [0] and number [1] of pixels, used for mean computation.
    float* gpu_mean_vector;
    cudaMalloc(&gpu_mean_vector, 2 * sizeof(float));
    cudaMemset(gpu_mean_vector, 0, 2 * sizeof(float));

    kernel_circular_mask<<<lblocks, lthreads, 0, stream>>>(mask, width, height, center_X, center_Y, radius);

    cudaXStreamSynchronize(stream);
    cudaCheckError();
}