#include <algorithm>
#include "texture_update.cuh"
#include "cuda_memory.cuh"

__global__ static void updateFloatSlice(ushort* frame, cudaSurfaceObject_t cuSurface, dim3 texDim)
{
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint index = y * texDim.x + x;

    if (x >= texDim.x || y >= texDim.y)
        return;

    surf2Dwrite(static_cast<uchar>(frame[index] >> 8), cuSurface, x << 2, y);
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

    surf2Dwrite(pix, cuSurface, x << 2, y);
}

void textureUpdate(cudaSurfaceObject_t cuSurface,
                   void* frame,
                   const holovibes::FrameDescriptor& fd,
                   const cudaStream_t stream)
{
    unsigned thread_width = std::min(32u, (unsigned)fd.width);
    unsigned thread_height = std::min(32u, (unsigned)fd.height);
    dim3 threads(thread_width, thread_height);

    const uint fd_width_div_32 = std::ceil(static_cast<float>(fd.width) / threads.x);
    const uint fd_height_div_32 = std::ceil(static_cast<float>(fd.height) / threads.y);
    dim3 blocks(fd_width_div_32, fd_height_div_32);

    if (fd.depth == 8)
    {
        updateComplexSlice<<<blocks, threads, 0, stream>>>(reinterpret_cast<cuComplex*>(frame),
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
