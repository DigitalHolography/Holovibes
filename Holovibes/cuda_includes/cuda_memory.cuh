/*! \file cuda_memory.cuh
 *
 * \brief Contains all the safe call wrappers around cuda calls.
 *
 * Not all cuda calls were included in this file, only the most used ones
 * To make a safe call, use our wrapper, if it's not in the list you can
 * either:
 * - Wrap it by adding it here
 * - Directly use 'cudaSafeCall' in your code
 *
 * The behavior in case of error can be found in the common.cuh file
 * Currently details about the file, line, error will be logged and the
 * programm will abort
 *
 * IMPORTANT NOTE : SAFECALLS ARE ONLY ENABLED IN DEBUG MODE
 * (you can again modify this behavior if you wish in the common.cuh file)
 */
#pragma once

#include "common.cuh"

/*! \brief Wrapper around cudaMalloc to handle errors
 *
 * This function uses the error handling from common.cuh (cudaSafeCall)
 * A program built in error WILL abort in case of error
 *
 * Only cuda malloc needs to be templated to avoid (void**) cast of the pointer
 * on the call
 *
 * \param devPtr The device pointer to allocate.
 * \param size Size in byte to allocate.
 *
 */
template <typename T>
void cudaXMalloc(T** devPtr, size_t size);

/*! \brief Wrapper around cudaMallocHost to handle errors
 *
 * This function uses the error handling from common.cuh (cudaSafeCall)
 * A program built in error WILL abort in case of error
 *
 * \param devPtr The device pointer to allocate.
 * \param size Size in byte to allocate.
 *
 */
template <typename T>
void cudaXMallocHost(T** devPtr, size_t size);

/*! \brief Wrapper around cudaMallocManaged to handle errors
 *
 * This function uses the error handling from common.cuh (cudaSafeCall)
 * A program built in error WILL abort in case of error
 *
 * \param devPtr The device pointer to allocate.
 * \param size Size in byte to allocate.
 *
 */
template <typename T>
void cudaXMallocManaged(T** devPtr, size_t size);

/*! \brief Wrapper around cudaMalloc for fast debugging
 *
 * \param devPtr The device pointer to allocate.
 * \param size Size in byte to allocate.
 *
 */
template <typename T>
cudaError_t cudaXRMalloc(T** devPtr, size_t size);

/*! \brief Wrapper around cudaMallocHost for fast debugging
 *
 * \param devPtr The device pointer to allocate.
 * \param size Size in byte to allocate.
 *
 */
template <typename T>
cudaError_t cudaXRMallocHost(T** devPtr, size_t size);

/*! \brief Wrapper around cudaMallocManaged for fast debugging
 *
 * \param devPtr The device pointer to allocate.
 * \param size Size in byte to allocate.
 *
 */
template <typename T>
cudaError_t cudaXRMallocManaged(T** devPtr, size_t size);

/*! \brief Wrapper around cudaMemcpy to handle errors
 *
 * This function uses the error handling from common.cuh (cudaSafeCall)
 * A program built in error WILL abort in case of error
 *
 * \param output Destination memory address.
 * \param input Source memory address.
 * \param count Size in bytes to copy.
 * \param kind Type of transfer.
 */
void cudaXMemcpy(void* output, const void* input, size_t count, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice);

/*! \brief Wrapper around cudaMemcpyAsync to handle errors
 *
 * This function uses the error handling from common.cuh (cudaSafeCall)
 * A program built in error WILL abort in case of error
 *
 * \param output Destination memory address.
 * \param input Source memory address.
 * \param count Size in bytes to copy.
 * \param kind Type of transfer.
 * \param stream Stream identifier.
 */
void cudaXMemcpyAsync(void* output, const void* input, size_t count, cudaMemcpyKind kind, const cudaStream_t stream);

/*! \brief Wrapper around cudaMemset to handle errors
 *
 * This function uses the error handling from common.cuh (cudaSafeCall)
 * A program built in error WILL abort in case of error
 *
 * \param output Destination memory address.
 * \param value Value to set for each byte of specified memory.
 * \param count Size in bytes to set.
 */
void cudaXMemset(void* output, int value, size_t count);

/*! \brief Wrapper around cudaMemsetAsync to handle errors
 *
 * This function uses the error handling from common.cuh (cudaSafeCall)
 * A program built in error WILL abort in case of error
 *
 * \param output Destination memory address.
 * \param value Value to set for each byte of specified memory.
 * \param count Size in bytes to set.
 */
void cudaXMemsetAsync(void* output, int value, size_t count, const cudaStream_t stream);

/*! \brief Wrapper around cudaFree to handle errors
 *
 * This function uses the error handling from common.cuh (cudaSafeCall)
 * A program built in error WILL abort in case of error
 *
 * \param devPtr Device pointer to memory to free
 */
void cudaXFree(void* devPtr);

/*! \brief Wrapper around cudaFreeHost to handle errors
 *
 * This function uses the error handling from common.cuh (cudaSafeCall)
 * A program built in error WILL abort in case of error
 *
 * \param devPtr Device pointer to memory to free
 */
void cudaXFreeHost(void* devPtr);

/*! \brief Wrapper around cudaStreamSynchronize to handle errors
 *
 * This function uses the error handling from common.cuh (cudaSafeCall)
 * A program built in error WILL abort in case of error
 *
 * \param stream The id of the stream to synchronize with host
 */
void cudaXStreamSynchronize(const cudaStream_t stream);

#include "cuda_memory.cuhxx"
