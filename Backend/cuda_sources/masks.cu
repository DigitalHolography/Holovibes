#include "masks.cuh"

#include "frame_desc.hh"

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
    }
}

__global__ void kernel_spectral_lens(cuFloatComplex* output,
                                     const int Nx,
                                     const int Ny,
                                     const float z,
                                     const float lambda,
                                     const float x_step,
                                     const float y_step)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < Nx && y < Ny)
    {
        float u_step = 1.0f / (Nx * x_step);
        float v_step = 1.0f / (Ny * y_step);

        float u = (x - (Nx >> 1)) * u_step;
        float v = (y - (Ny >> 1)) * v_step;

        float tmp = 1.0f - (lambda * lambda * (u * u + v * v));
        // Ensure positivity under sqrt.
        if (tmp < 0.0f)
            tmp = 0.0f;
        float phase = 2.0f * M_PI * z / lambda * sqrtf(tmp);

        // Store result as complex exponential.
        output[y * Nx + x] = make_cuFloatComplex(cosf(phase), sinf(phase));
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

        // If the point is inside the circle set the value to 1.
        output[index] = (distance_squared <= radius_squared);
    }
}

void get_circular_mask(float* output,
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

    kernel_circular_mask<<<lblocks, lthreads, 0, stream>>>(output, width, height, center_X, center_Y, radius);

    cudaXStreamSynchronize(stream);
    cudaCheckError();
}