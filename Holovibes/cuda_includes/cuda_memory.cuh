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

/*! \file cuda_memory.cuh
    \brief Contains all the safe call wrappers around cuda calls.

    Not all cuda calls were included in this file, only the most used ones
    To make a safe call, use our wrapper, if it's not in the list you can either:
     - Wrap it by adding it here
     - Directly use 'cudaSafeCall' in your code

    The behavior in case of error can be found in the Common.cuh file
    Currently details about the file, line, error will be logged and the programm will abort

    IMPORTANT NOTE : SAFECALLS ARE ONLY ENABLED IN DEBUG MODE
    (you can again modify this behavior if you wish in the Common.cuh file)
*/

#pragma once

/*! \brief Wrapper around cudaMalloc to handle errors
*
* This function uses the error handling from Common.cuh (cudaSafeCall)
* A program built in error WILL abort in case of error
*
* cudaXMalloc should be templated to avoid casting in (void**) (like cudaMalloc)
* Various attempts were made to template it (just add template or use a .cuhxx file)
* None worked, if you find a way, feel free to do it, the code would be cleaner
*
* \param devPtr The device pointer to allocate.
* \param size Size in byte to allocate.
*
*/
void cudaXMalloc(void** devPtr, size_t size);

/*! \brief Wrapper around cudaMallocHost to handle errors
*
* This function uses the error handling from Common.cuh (cudaSafeCall)
* A program built in error WILL abort in case of error
*
* \param devPtr The device pointer to allocate.
* \param size Size in byte to allocate.
*
*/
void cudaXMallocHost(void** devPtr, size_t size);

/*! \brief Wrapper around cudaMemcpy to handle errors
*
* This function uses the error handling from Common.cuh (cudaSafeCall)
* A program built in error WILL abort in case of error
*
* \param dst Destination memory address.
* \param src Source memory address.
* \param count Size in bytes to copy.
* \param kind Type of transfer.
*/
void cudaXMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice);

/*! \brief Wrapper around cudaMemcpyAsync to handle errors
*
* This function uses the error handling from Common.cuh (cudaSafeCall)
* A program built in error WILL abort in case of error
*
* \param dst Destination memory address.
* \param src Source memory address.
* \param count Size in bytes to copy.
* \param kind Type of transfer.
* \param stream Stream identifier.
*/
void cudaXMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0);

/*! \brief Wrapper around cudaMemset to handle errors
*
* This function uses the error handling from Common.cuh (cudaSafeCall)
* A program built in error WILL abort in case of error
*
* \param dst Destination memory address.
* \param value Value to set for each byte of specified memory.
* \param count Size in bytes to set.
*/
void cudaXMemset(void* devPtr, int  value, size_t count);

/*! \brief Wrapper around cudaMemsetAsync to handle errors
*
* This function uses the error handling from Common.cuh (cudaSafeCall)
* A program built in error WILL abort in case of error
*
* \param dst Destination memory address.
* \param value Value to set for each byte of specified memory.
* \param count Size in bytes to set.
*/
void cudaXMemsetAsync(void* devPtr, int  value, size_t count, cudaStream_t stream = 0);

/*! \brief Wrapper around cudaFree to handle errors
*
* This function uses the error handling from Common.cuh (cudaSafeCall)
* A program built in error WILL abort in case of error
*
* \param dst Device pointer to memory to free
*/
void cudaXFree(void* devPtr);

/*! \brief Wrapper around cudaFree to handle errors
*
* This function uses the error handling from Common.cuh (cudaSafeCall)
* A program built in error WILL abort in case of error
*
* \param dst Device pointer to memory to free
*/
void cudaXFreeHost(void* devPtr);