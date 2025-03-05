#include <algorithm>
#include "texture_update.cuh"
#include "cuda_memory.cuh"
#include "logger.hh"

__global__ static void update8BitSlice(uchar* frame, cudaSurfaceObject_t cuSurface, dim3 texDim)
{
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint index = y * texDim.x + x;

    if (x >= texDim.x || y >= texDim.y)
        return;

    surf2Dwrite(frame[index], cuSurface, x, y);
}

__global__ static void updateFloatSlice(ushort* frame, cudaSurfaceObject_t cuSurface, dim3 texDim)
{
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= texDim.x || y >= texDim.y)
        return;

    const uint index = y * texDim.x + x;

    uchar pixel8 = static_cast<uchar>(frame[index] >> 8);

    surf2Dwrite(pixel8, cuSurface, x, y);
}

__global__ static void updateComplexSlice(cuComplex* frame, cudaSurfaceObject_t cuSurface, dim3 texDim)
{
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= texDim.x || y >= texDim.y)
        return;

    const uint index = y * texDim.x + x;

    if (frame[index].x > 65535.f)
        frame[index].x = 65535.f;
    else if (frame[index].x < 0.f)
        frame[index].x = 0.f;

    if (frame[index].y > 65535.f)
        frame[index].y = 65535.f;
    else if (frame[index].y < 0.f)
        frame[index].y = 0.f;

    float pix = hypotf(frame[index].x, frame[index].y);

    float scale = 255.0f / (65535.0f * sqrtf(2.0f));
    unsigned char out_val = (unsigned char)(pix * scale);

    surf2Dwrite(out_val, cuSurface, x, y);
}

void textureUpdate(cudaSurfaceObject_t cuSurface,
                   void* frame,
                   const camera::FrameDescriptor& fd,
                   const cudaStream_t stream)
{
    unsigned thread_width = std::min(32u, (unsigned)fd.width);
    unsigned thread_height = std::min(32u, (unsigned)fd.height);
    dim3 threads(thread_width, thread_height);

    const uint fd_width_div_32 = std::ceil(static_cast<float>(fd.width) / threads.x);
    const uint fd_height_div_32 = std::ceil(static_cast<float>(fd.height) / threads.y);
    dim3 blocks(fd_width_div_32, fd_height_div_32);

    if (fd.depth == camera::PixelDepth::Complex)
    {
        updateComplexSlice<<<blocks, threads, 0, stream>>>(reinterpret_cast<cuComplex*>(frame),
                                                           cuSurface,
                                                           dim3(fd.width, fd.height));
    }
    else if (fd.depth == camera::PixelDepth::Bits8)
    {
        update8BitSlice<<<blocks, threads, 0, stream>>>(reinterpret_cast<uchar*>(frame),
                                                        cuSurface,
                                                        dim3(fd.width, fd.height));
    }
    else
    {
        updateFloatSlice<<<blocks, threads, 0, stream>>>(reinterpret_cast<ushort*>(frame),
                                                         cuSurface,
                                                         dim3(fd.width, fd.height));
    }

    cudaXStreamSynchronize(stream);
    cudaCheckError();
}
