/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#pragma once

#include "cuda_memory.cuh"

void cudaXMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind)
{
    cudaSafeCall(cudaMemcpy(dst, src, count, kind));
}

void cudaXMemcpyAsync(void* dst,
                      const void* src,
                      size_t count,
                      cudaMemcpyKind kind,
                      const cudaStream_t stream)
{
    cudaSafeCall(cudaMemcpyAsync(dst, src, count, kind, stream));
}

void cudaXMemset(void* devPtr, int value, size_t count)
{
    cudaSafeCall(cudaMemset(devPtr, value, count));
}

void cudaXMemsetAsync(void* devPtr,
                      int value,
                      size_t count,
                      const cudaStream_t stream)
{
    cudaSafeCall(cudaMemsetAsync(devPtr, value, count, stream));
}

void cudaXFree(void* devPtr) { cudaSafeCall(cudaFree(devPtr)); }

void cudaXFreeHost(void* devPtr) { cudaSafeCall(cudaFreeHost(devPtr)); }

void cudaXStreamSynchronize(const cudaStream_t stream,
                            const char* file,
                            const int line)
{
    std::cout << "File : " << file << " Line : " << line << std::endl;
    cudaSafeCall(cudaStreamSynchronize(stream));
}