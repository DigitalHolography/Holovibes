/*! \file map.cuh
 *
 *  \brief Optimized and generic map operation processed gpu side.
 *  The map operation applies a specified transformation or function to each element of an input buffer,
 *  producing a corresponding output buffer with the transformed values.
 */
#pragma once

#include <cuda_runtime.h>
#include "common.cuh"
using uint = unsigned int;
using ushort = unsigned short;

// The templated functions are only compilable via nvcc. Neccesary for compilation.
#if __NVCC__

/*! \brief Anonymous namespace to contain internal private functions.
 *  They must not be called from outisde this file.
 */
namespace
{
/*! \brief Basic kernel to apply a map operation.
 *  Apply `func` parameter on every pixel of the `input` buffer and store it in the `output` buffer.
 *
 *  \param[out] output The output to store the pixel after transformation.
 *  \param[in] input The input buffer to get the pixels.
 *  \param[in] size The size of an image.
 *  \param[in] func The function to apply on each buffer.
 */
template <typename O, typename I, typename FUNC>
__global__ static void kernel_map_generic(O* const output, const I* const input, const size_t size, const FUNC func)
{
    const size_t index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < size)
        output[index] = func(input[index]);
}

/*! \brief Wrapper to call `kernel_map_generic`
 *
 *  This function calls the kernel in the most optimized way.
 *
 *  \param[out] output The output to store the pixel after transformation.
 *  \param[in] input The input buffer to get the pixels.
 *  \param[in] size The size of an image.
 *  \param[in] func The function to apply on each buffer.
 *  \param[in] stream The CUDA stream to parallelize the computations.
 */
template <typename O, typename I, typename FUNC>
static void
map_generic_caller(O* const output, const I* const input, const size_t size, const FUNC func, const cudaStream_t stream)
{
    // Using 64 threads per blocks is more efficient than using the max number of threads supported
    constexpr uint threads = 64;
    const uint blocks = map_blocks_to_problem(size, threads);

    kernel_map_generic<<<blocks, threads, 0, stream>>>(output, input, size, func);
    cudaCheckError();
}

/*! \brief Map input to output throughout a mapping function on 4 dimension vector.
 *
 *  This operation only works with cuda vector types (float4, ushort4...).
 *  Thus, the size must be divisble by 4.
 *
 *  If users wish to map an array of custom types. This can be done by creating
 *  a structure of 4 element.
 *  (i.e class myClass comes the vectorized type myClass4 { myClass x; myClass y; myClass z; myClass w };)
 *
 *  This function is non static because the created lambda has a closure containing
 *  the func parameter. This forbids from making this function static.
 *
 *  \param[out] output The output to store the pixel after transformation.
 *  \param[in] input The input buffer to get the pixels.
 *  \param[in] size The size of an image.
 *  \param[in] func The function to apply on each buffer.
 *  \param[in] stream The CUDA stream to parallelize the computations.
 */
template <typename T, typename FUNC>
void map_vector(T* const output, const T* const input, const size_t size, const FUNC func, const cudaStream_t stream)
{
    CHECK(size % 4 == 0, "Map vectorize requires a size divisible by 4.");

    // Instead of one thread per pixel, in vectorize map each thread works on 4 pixels.
    // The 4 pixels are grouped in the vectorized type (T)
    const auto vectorized_lambda = [func] __device__(const T in) -> T {
        return T{func(in.x), func(in.y), func(in.z), func(in.w)};
    };

    // Call generic map with the new vectorized lambda and an updated size.
    map_generic_caller(output, input, size / 4, vectorized_lambda, stream);
}
} // namespace

/*! \brief Map input to output throughout a mapping function.
 *
 *  This function is the generic map operation for any types. It means that it is not the most optimized map operation.
 *  For instance, this map operation cannot be vectorized because only some types (float, ushort, uint...) can be
 *  vectorized. Moreover, this operation works on any sizes.
 *
 *  \param[out] output The output to store the pixel after transformation.
 *  \param[in] input The input buffer to get the pixels.
 *  \param[in] size The size of an image.
 *  \param[in] func The function to apply on each buffer.
 *  \param[in] stream The CUDA stream to parallelize the computations.
 */
template <typename O, typename I, typename FUNC>
void map_generic(O* const output, const I* const input, const size_t size, const FUNC func, const cudaStream_t stream)
{
    map_generic_caller(output, input, size, func, stream);
}

/*! \brief Map input (float) to output (float) throughout a mapping function.
 *
 *  This function is the specialized map operation for float arrays.
 *  It means that it is the most optimized map operation for float arrays.
 *  When possible (if size is divisible by four) the vectorized map function is
 *  called. Otherwise, the generic (any types, any size) map function is called.
 *
 *  This function overloads the templated generic function with float.
 *
 *  \param[out] output The output to store the pixel after transformation.
 *  \param[in] input The input buffer to get the pixels.
 *  \param[in] size The size of an image.
 *  \param[in] func The function to apply on each buffer.
 *  \param[in] stream The CUDA stream to parallelize the computations.
 */
template <typename FUNC>
void map_generic(
    float* const output, const float* const input, const size_t size, const FUNC func, const cudaStream_t stream)
{
    if (size % 4 == 0)
        map_vector<float4>(reinterpret_cast<float4* const>(output),
                           reinterpret_cast<const float4* const>(input),
                           size,
                           func,
                           stream);
    else
        map_generic_caller(output, input, size, func, stream);
}

/*! \brief Map input (ushort) to output (ushort) throughout a mapping function.
 *
 *  This function is the specialized map operation for float arrays.
 *  It means that it is the most optimized map operation for float arrays.
 *  When possible (if size is divisible by four) the vectorized map function is
 *  called. Otherwise, the generic (any types, any size) map function is called.
 *
 *  This function overloads the templated generic function with float.
 *
 *  \param[out] output The output to store the pixel after transformation.
 *  \param[in] input The input buffer to get the pixels.
 *  \param[in] size The size of an image.
 *  \param[in] func The function to apply on each buffer.
 *  \param[in] stream The CUDA stream to parallelize the computations.
 */
template <typename FUNC>
void map_generic(
    ushort* const output, const ushort* const input, const size_t size, const FUNC func, const cudaStream_t stream)
{
    if (size % 4 == 0)
        map_vector<ushort4>(reinterpret_cast<ushort4* const>(output),
                            reinterpret_cast<const ushort4* const>(input),
                            size,
                            func,
                            stream);
    else
        map_generic_caller(output, input, size, func, stream);
}

#endif

/*! \brief Apply log10 on every pixel of the input (float array).
 *
 *  \param[out] output The output buffer after log10 application.
 *  \param[in] input The input buffer to get the images.
 *  \param[in] size The size of an image.
 *  \param[in] stream The CUDA stream to parallelize the computations.
 */
void map_log10(float* const output, const float* const input, const size_t size, const cudaStream_t stream);

/*! \brief Divide every pixel of a float array by a value.
 *
 *  \param[out] output The output buffer after division application.
 *  \param[in] input The input buffer to get the images.
 *  \param[in] size The size of an image.
 *  \param[in] value The value to use for the division.
 *  \param[in] stream The CUDA stream to parallelize the computations.
 */
void map_divide(
    float* const output, const float* const input, const size_t size, const float value, const cudaStream_t stream);

/*! \brief Multiply every pixel of a float array by a value.
 *
 *  \param[out] output The output buffer after multiplication application.
 *  \param[in] input The input buffer to get the images.
 *  \param[in] size The size of an image.
 *  \param[in] value The value to use for the multiplication.
 *  \param[in] stream The CUDA stream to parallelize the computations.
 */
void map_multiply(
    float* const output, const float* const input, const size_t size, const float value, const cudaStream_t stream);