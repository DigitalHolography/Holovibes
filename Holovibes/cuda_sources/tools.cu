#include "tools.cuh"

#include <cassert>

#include "common.cuh"
#include "cuda_tools/unique_ptr.hh"
#include "cuda_tools/cufft_handle.hh"
#include "cuda_memory.cuh"
#include "frame_desc.hh"
#include "logger.hh"
#include "tools_compute.cuh"
#include "tools_unwrap.cuh"

using camera::FrameDescriptor;
using namespace holovibes;
using cuda_tools::CudaUniquePtr;
using cuda_tools::CufftHandle;

__global__ void kernel_complex_to_modulus(float* output, const cuComplex* input, const uint size)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
        output[index] = hypotf(input[index].x, input[index].y);
}

void frame_memcpy(
    float* output, const float* input, const units::RectFd& zone, const uint input_width, const cudaStream_t stream)
{
    const float* zone_ptr = input + (zone.topLeft().y() * input_width + zone.topLeft().x());
    cudaSafeCall(cudaMemcpy2DAsync(output,
                                   zone.width() * sizeof(float),
                                   zone_ptr,
                                   input_width * sizeof(float),
                                   zone.width() * sizeof(float),
                                   zone.height(),
                                   cudaMemcpyDeviceToDevice,
                                   stream));
}

/*! \brief CUDA Kernel to perform circ_shift computations in parallel.
 *
 *  \param[out] output The buffer to store the output image.
 *  \param[in] input The input image.
 *  \param[in] width The width of the image.
 *  \param[in] height The height of the image.
 *  \param[in] shift_x The x point to shift.
 *  \param[in] shift_y The y point to shift.
 */
__global__ void circ_shift_kernel(float* output, const float* input, int width, int height, int shift_x, int shift_y)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        // Computing new coordinates after shift application.
        int new_x = (x + shift_x) % width;
        int new_y = (y + shift_y) % height;

        // Avoid negative shifts.
        new_x += width * (new_x < 0);
        new_y += height * (new_y < 0);

        // Copy of the pixel at the new position.
        output[new_y * width + new_x] = input[y * width + x];
    }
}

void circ_shift(float* output, float* input, uint width, uint height, int shift_x, int shift_y, cudaStream_t stream)
{
    uint threads_2d = get_max_threads_2d();
    dim3 lthreads(threads_2d, threads_2d);
    dim3 lblocks(1 + (width - 1) / threads_2d, 1 + (height - 1) / threads_2d);

    circ_shift_kernel<<<lblocks, lthreads, 0, stream>>>(output, input, width, height, shift_x, shift_y);

    cudaCheckError();
}