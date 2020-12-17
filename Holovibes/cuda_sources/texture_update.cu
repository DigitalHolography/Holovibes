/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#include <algorithm>
#include "texture_update.cuh"

__global__ static void
updateFloatSlice(ushort* frame, cudaSurfaceObject_t cuSurface, dim3 texDim)
{
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint index = y * texDim.x + x;

    surf2Dwrite(static_cast<uchar>(frame[index] >> 8), cuSurface, x << 2, y);
}

__global__ static void
updateComplexSlice(cuComplex* frame, cudaSurfaceObject_t cuSurface, dim3 texDim)
{
    const uint xId = blockIdx.x * blockDim.x + threadIdx.x;
    const uint yId = blockIdx.y * blockDim.y + threadIdx.y;
    const uint index = yId * texDim.x + xId;

    if (frame[index].x > 65535.f)
        frame[index].x = 65535.f;
    else if (frame[index].x < 0.f)
        frame[index].x = 0.f;

    if (frame[index].y > 65535.f)
        frame[index].y = 65535.f;
    else if (frame[index].y < 0.f)
        frame[index].y = 0.f;
    float pix = hypotf(frame[index].x, frame[index].y);

    surf2Dwrite(pix, cuSurface, xId << 2, yId);
}

void textureUpdate(cudaSurfaceObject_t cuSurface,
                   void* frame,
                   const camera::FrameDescriptor& fd,
                   const cudaStream_t stream)
{

    const uint fd_width_div_32 = std::max(1u, (unsigned)fd.width / 32u);
    const uint fd_height_div_32 = std::max(1u, (unsigned)fd.height / 32u);
    dim3 blocks(fd_width_div_32, fd_height_div_32);

    unsigned thread_width = std::min(32u, (unsigned)fd.width);
    unsigned thread_height = std::min(32u, (unsigned)fd.height);
    dim3 threads(thread_width, thread_height);

    if (fd.depth == 8)
    {
        updateComplexSlice<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<cuComplex*>(frame),
            cuSurface,
            dim3(fd.width, fd.height));
    }
    else
    {
        updateFloatSlice<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<ushort*>(frame),
            cuSurface,
            dim3(fd.width, fd.height));
    }

    cudaStreamSynchronize(stream);
    cudaCheckError();
}
