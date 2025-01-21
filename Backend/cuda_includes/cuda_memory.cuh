/*! \file cuda_memory.cuh
 *
 *  \brief Contains all the safe call wrappers around cuda calls.
 *
 *  Not all cuda calls were included in this file, only the most used ones.
 *  To make a safe call, use our wrapper, if it's not in the list you can
 *  either:
 *  - Wrap it by adding it here
 *  - Directly use 'cudaSafeCall' in your code
 *
 *  The behavior in case of error can be found in the common.cuh file.
 *  Currently details about the file, line, error will be logged and the
 *  programm will abort.
 *
 *  IMPORTANT NOTE : SAFECALLS ARE ONLY ENABLED IN DEBUG MODE
 *  (you can again modify this behavior if you wish in the common.cuh file)
 */
#pragma once

#include "common.cuh"
#include "logger.hh"

/*! \brief Wrapper around cudaMallocManaged for fast debugging.
 *
 *  \param[in] ptr The device pointer to allocate.
 *  \param[in] size Size in byte to allocate.
 */
template <typename T>
cudaError_t cudaXRMallocManaged(T** ptr, size_t size);

/*! \brief Wrapper around cudaMallocManaged to handle errors.
 *
 *  This function uses the error handling from common.cuh (cudaSafeCall).
 *  A program built in error WILL abort in case of error.
 *
 *  \param[in] ptr The device pointer to allocate.
 *  \param[in] size Size in byte to allocate.
 */
template <typename T>
void cudaXMallocManaged(T** ptr, size_t size)
{
    cudaSafeCall(cudaXRMallocManaged(ptr, size));
}

/*! \brief Wrapper around cudaMalloc for fast debugging.
 *
 *  \param[in] ptr The device pointer to allocate.
 *  \param[in] size Size in byte to allocate.
 */
template <typename T>
cudaError_t cudaXRMalloc(T** ptr, size_t size)
{
    LOG_DEBUG("Allocate {:.3f} Gib on Device", static_cast<float>(size) / (1024 * 1024 * 1024));
    return cudaMalloc(ptr, size);
}

/*! \brief Wrapper around cudaMalloc to handle errors.
 *
 *  This function uses the error handling from common.cuh (cudaSafeCall).
 *  A program built in error WILL abort in case of error.
 *
 *  Only cuda malloc needs to be templated to avoid (void**) cast of the pointer
 *  on the call.
 *
 *  \param[in] ptr The device pointer to allocate.
 *  \param[in] size Size in byte to allocate.
 */
template <typename T>
void cudaXMalloc(T** ptr, size_t size)
{
    cudaSafeCall(cudaXRMalloc(ptr, size));
}

/*! \brief Wrapper around cudaMallocHost for fast debugging.
 *
 *  \param[in] ptr The device pointer to allocate.
 *  \param[in] size Size in byte to allocate.
 */
template <typename T>
cudaError_t cudaXRMallocHost(T** ptr, size_t size)
{
    LOG_DEBUG("Allocate {:.3f} Gib on Host", static_cast<float>(size) / (1024 * 1024 * 1024));
    return cudaMallocHost(ptr, size);
}

/*! \brief Wrapper around cudaMallocHost to handle errors.
 *
 *  This function uses the error handling from common.cuh (cudaSafeCall).
 *  A program built in error WILL abort in case of error.
 *
 *  \param[in] ptr The device pointer to allocate.
 *  \param[in] size Size in byte to allocate.
 */
template <typename T>
void cudaXMallocHost(T** ptr, size_t size)
{
    cudaSafeCall(cudaXRMallocHost(ptr, size));
}

/*! \brief Wrapper around cudaMallocManaged for fast debugging.
 *
 *  \param[in] ptr The device pointer to allocate.
 *  \param[in] size Size in byte to allocate.
 */
template <typename T>
cudaError_t cudaXRMallocManaged(T** ptr, size_t size)
{
    LOG_DEBUG("Allocate {:.3f} Gib on Device w/ Managed", static_cast<float>(size) / (1024 * 1024 * 1024));
    return cudaMallocManaged(ptr, size);
}

/*! \brief Get the size allocated on GPU of the given `ptr`.
 *  A call to `cuMemGetAddressRange` is made.
 *
 *  \param[in] ptr The device pointer to allocate.
 *  \param[in] size Size in byte to allocate.
 */
template <typename T>
inline size_t cudaGetAllocatedSize(T* ptr)
{
    CUdeviceptr pbase;
    size_t psize;
    cuMemGetAddressRange(&pbase, &psize, (CUdeviceptr)ptr);
    return psize;
}

/*! \brief Wrapper around cudaMemcpy to handle errors.
 *
 *  This function uses the error handling from common.cuh (cudaSafeCall).
 *  A program built in error WILL abort in case of error.
 *
 *  \param[out] output Destination memory address.
 *  \param[in] input Source memory address.
 *  \param[in] count Size in bytes to copy.
 *  \param[in] kind Type of transfer.
 */
inline void cudaXMemcpy(void* output, const void* input, size_t count, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice)
{
    cudaSafeCall(cudaMemcpy(output, input, count, kind));
}

/*! \brief Wrapper around cudaMemcpyAsync to handle errors.
 *
 *  This function uses the error handling from common.cuh (cudaSafeCall).
 *  A program built in error WILL abort in case of error.
 *
 *  \param[out] output Destination memory address.
 *  \param[in] input Source memory address.
 *  \param[in] count Size in bytes to copy.
 *  \param[in] kind Type of transfer.
 *  \param[in] stream Stream identifier.
 */
inline void
cudaXMemcpyAsync(void* output, const void* input, size_t count, cudaMemcpyKind kind, const cudaStream_t stream = 0)
{
    cudaSafeCall(cudaMemcpyAsync(output, input, count, kind, stream));
}

/*! \brief Wrapper around cudaMemset to handle errors.
 *
 *  This function uses the error handling from common.cuh (cudaSafeCall).
 *  A program built in error WILL abort in case of error.
 *
 *  \param[out] output Destination memory address.
 *  \param[in] value Value to set for each byte of specified memory.
 *  \param[in] count Size in bytes to set.
 */
inline void cudaXMemset(void* output, int value, size_t count) { cudaSafeCall(cudaMemset(output, value, count)); }

/*! \brief Wrapper around cudaMemsetAsync to handle errors.
 *
 *  This function uses the error handling from common.cuh (cudaSafeCall).
 *  A program built in error WILL abort in case of error.
 *
 *  \param[out] output Destination memory address.
 *  \param[in] value Value to set for each byte of specified memory.
 *  \param[in] count Size in bytes to set.
 */
inline void cudaXMemsetAsync(void* output, int value, size_t count, const cudaStream_t stream)
{
    cudaSafeCall(cudaMemsetAsync(output, value, count, stream));
}

/*! \brief Wrapper around cudaFree to handle errors.
 *
 *  This function uses the error handling from common.cuh (cudaSafeCall).
 *  A program built in error WILL abort in case of error.
 *
 *  \param[in] ptr Device pointer to memory to free.
 */
inline void cudaXFree(void* ptr) { cudaSafeCall(cudaFree(ptr)); }

/*! \brief Wrapper around cudaFreeHost to handle errors.
 *
 *  This function uses the error handling from common.cuh (cudaSafeCall).
 *  A program built in error WILL abort in case of error.
 *
 *  \param[in] ptr Device pointer to memory to free.
 */
inline void cudaXFreeHost(void* ptr) { cudaSafeCall(cudaFreeHost(ptr)); }

/*! \brief Wrapper around cudaStreamSynchronize to handle errors.
 *
 *  This function uses the error handling from common.cuh (cudaSafeCall).
 *  A program built in error WILL abort in case of error.
 *
 *  \param[in] stream The id of the stream to synchronize with host.
 */
inline void cudaXStreamSynchronize(const cudaStream_t stream) { cudaSafeCall(cudaStreamSynchronize(stream)); }