/* **************************************************************************** */
/*                       ,,                     ,,  ,,                          */
/* `7MMF'  `7MMF'       `7MM       `7MMF'   `7MF'db *MM                         */
/*   MM      MM           MM         `MA     ,V      MM                         */
/*   MM      MM  ,pW"Wq.  MM  ,pW"Wq. VM:   ,V `7MM  MM,dMMb.   .gP"Ya  ,pP"Ybd */
/*   MMmmmmmmMM 6W'   `Wb MM 6W'   `Wb MM.  M'   MM  MM    `Mb ,M'   Yb 8I   `" */
/*   MM      MM 8M     M8 MM 8M     M8 `MM A'    MM  MM     M8 8M"""""" `YMMMa. */
/*   MM      MM YA.   ,A9 MM YA.   ,A9  :MM;     MM  MM.   ,M9 YM.    , L.   I8 */
/* .JMML.  .JMML.`Ybmd9'.JMML.`Ybmd9'    VF    .JMML.P^YbmdP'   `Mbmmd' M9mmmP' */
/*                                                                              */
/* **************************************************************************** */

#pragma once

#include "cuda_memory.cuh"
#include "Common.cuh"

void cudaXMalloc(void** devPtr, size_t size)
{
    cudaSafeCall(cudaMalloc(devPtr, size));
}

void cudaXMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind)
{
    cudaSafeCall(cudaMemcpy(dst, src, count, kind));
}

void cudaXMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream)
{
    cudaSafeCall(cudaMemcpyAsync(dst, src, count, kind, stream));
}

void cudaXMemset(void* devPtr, int  value, size_t count)
{
    cudaSafeCall(cudaMemset(devPtr, value, count));
}

void cudaXMemsetAsync(void* devPtr, int  value, size_t count, cudaStream_t stream)
{
    cudaSafeCall(cudaMemsetAsync(devPtr, value, count, stream));
}

void cudaXFree(void* devPtr)
{
    cudaSafeCall(cudaFree(devPtr));
}