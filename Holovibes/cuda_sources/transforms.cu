#include "transforms.cuh"

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