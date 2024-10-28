#include "masks.cuh"

using camera::FrameDescriptor;

__global__ void kernel_quadratic_lens(
    cuComplex* output, const uint lens_side_size, const float lambda, const float dist, const float pixel_size)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    const float c = M_PI / (lambda * dist);
    const float dx = pixel_size * 1.0e-6f;
    const float dy = dx;
    float x, y;
    uint i, j;
    float csquare;

    if (index < lens_side_size * lens_side_size)
    {
        i = index % lens_side_size;
        j = index / lens_side_size;
        x = (i - static_cast<float>(lens_side_size >> 1)) * dx;
        y = (j - static_cast<float>(lens_side_size >> 1)) * dy;

        csquare = c * (x * x + y * y);
        output[index].x = cosf(csquare);
        output[index].y = sinf(csquare);
        // output[index].x = ((float)i) / (float)lens_side_size * 2 - 1;
        // output[index].y = ((float)j) / (float)lens_side_size * 2 - 1;
    }
}

__global__ void kernel_spectral_lens(
    cuComplex* output, const uint lens_side_size, const float lambda, const float distance, const float pixel_size)
{
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint j = blockIdx.y * blockDim.y + threadIdx.y;
    const uint index = j * blockDim.x * gridDim.x + i;
    const float c = M_2PI * distance / lambda;
    const float dx = pixel_size * 1.0e-6f;
    const float dy = dx;
    const float du = 1 / ((static_cast<float>(lens_side_size)) * dx);
    const float dv = 1 / ((static_cast<float>(lens_side_size)) * dy);
    const float u = (i - static_cast<float>(lrintf(static_cast<float>(lens_side_size >> 1)))) * du;
    const float v = (j - static_cast<float>(lrintf(static_cast<float>(lens_side_size >> 1)))) * dv;

    if (index < lens_side_size * lens_side_size)
    {
        const float lambda2 = lambda * lambda;
        const float csquare = c * sqrtf(abs(1.0f - lambda2 * u * u - lambda2 * v * v));
        output[index].x = cosf(csquare);
        output[index].y = sinf(csquare);
    }
}

__global__ void
kernel_circular_mask(float* output, short width, short height, float center_X, float center_Y, float radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int index = y * width + x;

    if (x < width && y < height)
    {
        float distance_squared = (x - center_X) * (x - center_X) + (y - center_Y) * (y - center_Y);
        float radius_squared = radius * radius;

        if (distance_squared > radius_squared)
            output[index] = 0.0f;
        else
        {
            output[index] = 1.0f;
        }
    }
}

void get_circular_mask(float* mask,
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

    // Allocating memory on GPU to compute the sum [0] and number [1] of pixels, used for mean computation.
    float* gpu_mean_vector;
    cudaMalloc(&gpu_mean_vector, 2 * sizeof(float));
    cudaMemset(gpu_mean_vector, 0, 2 * sizeof(float));

    kernel_circular_mask<<<lblocks, lthreads, 0, stream>>>(mask, width, height, center_X, center_Y, radius);

    cudaXStreamSynchronize(stream);
    cudaCheckError();
}