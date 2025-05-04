/*! \file
 *
 * \brief Utils functions for Analysis functions
 */
#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

typedef unsigned int uint;

/*!
 * \brief Finds the index of the maximum element in an array using Thrust library.
 *
 * This function takes an array of floats and its size, and returns the index of the maximum element in the array.
 * It utilizes the Thrust library to perform the operation on the GPU.
 *
 * \param input Pointer to the input array of floats.
 * \param size The number of elements in the input array.
 * \return The index of the maximum element in the input array.
 */
int find_max_thrust(float* input, const size_t size);

/*!
 * \brief Finds the index of the minimum element in an array using Thrust library.
 *
 * This function takes an array of floats and its size, and returns the index of the minimum element in the array.
 * It utilizes the Thrust library to perform the operation on the GPU.
 *
 * \param input Pointer to the input array of floats.
 * \param size The number of elements in the input array.
 * \return The index of the minimum element in the input array.
 */
int find_min_thrust(float* input, const size_t size);

/*!
 * \brief Launches a CUDA kernel to generate a normalized list.
 *
 * This function configures and launches the `kernel_normalized_list` on the GPU.
 *
 * \param [out] output Pointer to the output array on the device memory.
 * \param [in] lim The limit value to subtract from each index.
 * \param [in] size The total number of elements to compute.
 * \param [in] stream CUDA stream for asynchronous kernel execution.
 */
void normalized_list(float* output, int lim, int size, cudaStream_t stream);

/*!
 * \brief Launches a CUDA kernel to compute the nth derivative of a Gaussian function for an array of inputs.
 *
 * This function launches `kernel_comp_dgaussian` to compute the nth derivative of a Gaussian function for each element
 * in the input array. The results are stored in the output array.
 *
 * \param [out] output Pointer to the output array in device memory where results are stored.
 * \param [in] input Pointer to the input array in device memory containing the x values.
 * \param [in] input_size The number of elements in the input array.
 * \param [in] sigma The standard deviation (\( \sigma > 0 \)) of the Gaussian function.
 * \param [in] n The order of the derivative to compute (\( n \geq 0 \)).
 * \param [in] stream The CUDA stream to be used for asynchronous execution.
 *
 * \note Ensure that the `output` and `input` pointers reference valid, allocated memory in the device.
 *       The sizes of both arrays must be at least `input_size` elements.
 */
void comp_dgaussian(float* output, float* input, size_t input_size, float sigma, int n, cudaStream_t stream);

/*!
 * \brief Applies a circular diaphragm mask to an output array using a CUDA kernel.
 *
 * This function configures and launches a CUDA kernel to apply a circular diaphragm mask to an output array.
 * The mask is centered at (center_X, center_Y) with a specified radius.
 * Points outside the circle are set to zero, while points inside the circle remain unchanged.
 * The function uses the provided CUDA stream for asynchronous execution.
 *
 * \param [in,out] output Pointer to the output array where the mask will be applied.
 * \param [in] center_X The x-coordinate of the center of the circular mask.
 * \param [in] center_Y The y-coordinate of the center of the circular mask.
 * \param [in] radius The radius of the circular mask.
 * \param [in] width The width of the output array.
 * \param [in] height The height of the output array.
 * \param [in] stream The CUDA stream to use for the kernel launch.
 *
 * \note The function configures the kernel launch parameters based on the dimensions of the output array
 *       and calls `cudaCheckError()` to check for any CUDA errors after the kernel launch.
 */
void apply_diaphragm_mask(float* output,
                          const float center_X,
                          const float center_Y,
                          const float radius,
                          const short width,
                          const short height,
                          const cudaStream_t stream);

/*!
 * \brief Computes the eigenvalues of 2x2 symmetric matrices stored in an array using a CUDA kernel.
 *
 * This function launches a CUDA kernel to compute the eigenvalues of 2x2 symmetric matrices stored in an array.
 * It configures the kernel launch parameters based on the size of the input array and uses the provided CUDA stream
 * for asynchronous execution. The eigenvalues are stored in the provided output arrays.
 *
 * \param H Pointer to the input array containing the elements of the 2x2 symmetric matrices.
 *          The array is expected to have the following layout: [a1, a2, ..., an, b1, b2, ..., bn, d1, d2, ..., dn],
 *          where each matrix is represented by (a, b, d).
 * \param size The number of 2x2 matrices in the input array.
 * \param lambda1 Pointer to the output array where the first eigenvalue of each matrix will be stored.
 * \param lambda2 Pointer to the output array where the second eigenvalue of each matrix will be stored.
 * \param stream The CUDA stream to use for the kernel launch.
 *
 * \note The function assumes that the input array H contains the elements of the matrices in the specified layout.
 *       The eigenvalues are computed using the characteristic equation of the 2x2 symmetric matrices.
 *       The function calls `cudaCheckError()` to check for any CUDA errors after the kernel launch.
 */
void compute_eigen_values(float* H, int size, float* lambda1, float* lambda2, cudaStream_t stream);

/*!
 * \brief Computes a circular mask and applies it to an output array using a CUDA kernel.
 *
 * This function configures and launches a CUDA kernel to compute a circular mask and apply it to an output array.
 * The mask is centered at (center_X, center_Y) with a specified radius.
 * Points inside the circle are set to 1, while points outside the circle are set to 0.
 * The function uses the provided CUDA stream for asynchronous execution.
 *
 * \param [out] output Pointer to the output array where the mask will be applied.
 * \param [in] center_X The x-coordinate of the center of the circular mask.
 * \param [in] center_Y The y-coordinate of the center of the circular mask.
 * \param [in] radius The radius of the circular mask.
 * \param [in] width The width of the output array.
 * \param [in] height The height of the output array.
 * \param [in] stream The CUDA stream to use for the kernel launch.
 *
 * \note The function configures the kernel launch parameters based on the dimensions of the output array
 *       and calls `cudaCheckError()` to check for any CUDA errors after the kernel launch.
 */
void compute_circle_mask(float* output,
                         const float center_X,
                         const float center_Y,
                         const float radius,
                         const short width,
                         const short height,
                         const cudaStream_t stream);

/*!
 * \brief Applies a mask to an output array by performing element-wise multiplication with an input array using a CUDA
 * kernel.
 *
 * This function configures and launches a CUDA kernel to apply a mask to an output array by performing element-wise
 * multiplication with the corresponding elements of an input array. The function uses the provided CUDA stream for
 * asynchronous execution.
 *
 * \param [in,out] output Pointer to the output array where the mask will be applied.
 * \param [in] input Pointer to the input array containing the mask values.
 * \param [in] width The width of the output and input arrays.
 * \param [in] height The height of the output and input arrays.
 * \param [in] stream The CUDA stream to use for the kernel launch.
 *
 * \note The function configures the kernel launch parameters based on the dimensions of the output array
 *       and calls `cudaCheckError()` to check for any CUDA errors after the kernel launch.
 */
void apply_mask_and(
    float* output, const float* input, const short width, const short height, const cudaStream_t stream);

/*!
 * \brief Applies a mask to an output array by performing an element-wise logical OR operation with an input array using
 * a CUDA kernel.
 *
 * This function configures and launches a CUDA kernel to apply a mask to an output array by performing an element-wise
 * logical OR operation with the corresponding elements of an input array. If an element in the input array is non-zero,
 * the corresponding element in the output array is set to 1. Otherwise, the element in the output array remains
 * unchanged. The function uses the provided CUDA stream for asynchronous execution.
 *
 * \param [in,out] output Pointer to the output array where the mask will be applied.
 * \param [in] input Pointer to the input array containing the mask values.
 * \param [in] width The width of the output and input arrays.
 * \param [in] height The height of the output and input arrays.
 * \param [in] stream The CUDA stream to use for the kernel launch.
 *
 * \note The function configures the kernel launch parameters based on the dimensions of the output array
 *       and calls `cudaCheckError()` to check for any CUDA errors after the kernel launch.
 */
void apply_mask_or(float* output, const float* input, const short width, const short height, const cudaStream_t stream);

/*!
 * \brief Computes a Gaussian kernel and normalizes it using CUDA.
 *
 * This function computes a Gaussian kernel of a specified standard deviation (sigma) and normalizes it.
 * It allocates device memory for the sum of the kernel values, initializes it, and launches a CUDA kernel
 * to compute the Gaussian values. After computing the values, it normalizes the kernel using the computed sum.
 * The function uses the provided CUDA stream for asynchronous execution.
 *
 * \param [out] output Pointer to the output array where the Gaussian kernel values will be stored.
 * \param [in] sigma The standard deviation of the Gaussian kernel.
 * \param [in] stream The CUDA stream to use for the kernel launch and memory operations.
 *
 * \note The function calculates the kernel size based on the standard deviation, allocates and initializes
 *       device memory for the sum, launches the kernel to compute the Gaussian values, normalizes the kernel,
 *       and frees the device memory for the sum. It also synchronizes the stream to ensure completion before
 *       freeing the memory.
 */
void compute_gauss_kernel(float* output, float sigma, cudaStream_t stream);

/*!
 * \brief Counts the number of non-zero elements in an input array using a CUDA kernel.
 *
 * This function counts the number of non-zero elements in an input array using a CUDA kernel.
 * It allocates device memory for the input array and the count variable, copies the input array to the device,
 * initializes the count variable, launches the CUDA kernel to count the non-zero elements, and copies the result back
 * to the host. The function uses the provided CUDA stream for asynchronous execution.
 *
 * \param [in] input Pointer to the input array containing the elements to be counted.
 * \param [in] rows The number of rows in the input array.
 * \param [in] cols The number of columns in the input array.
 * \param [in] stream The CUDA stream to use for the kernel launch and memory operations.
 * \return The total number of non-zero elements in the input array.
 *
 * \note The function allocates and frees device memory for the input array and the count variable.
 *       It also synchronizes the stream to ensure completion before freeing the memory.
 */
int count_non_zero(const float* const input, const int rows, const int cols, cudaStream_t stream);

/*!
 * \brief Divides each element of an input array by the corresponding element of a denominator array in place using a
 * CUDA kernel.
 *
 * This function configures and launches a CUDA kernel to divide each element of an input array by the corresponding
 * element of a denominator array. The operation is performed in place, meaning the input array is modified directly.
 * The function uses the provided CUDA stream for asynchronous execution.
 *
 * \param [in,out] input_output Pointer to the input array where the division will be performed in place.
 * \param [in] denominator Pointer to the array containing the denominator values.
 * \param [in] size The number of elements in the input and denominator arrays.
 * \param [in] stream The CUDA stream to use for the kernel launch.
 *
 * \note The function configures the kernel launch parameters based on the size of the input array
 *       and calls `cudaCheckError()` to check for any CUDA errors after the kernel launch.
 */
void divide_frames_inplace(float* const input_output,
                           const float* const denominator,
                           const uint size,
                           cudaStream_t stream);

/*!
 * \brief Normalizes the elements of an array to a specified range using a CUDA kernel.
 *
 * This function normalizes the elements of an input array to a specified range [min_range, max_range].
 * It first uses the Thrust library to find the minimum and maximum values in the array on the device.
 * Then, it launches a CUDA kernel to perform the normalization. The function uses the provided CUDA stream
 * for asynchronous execution.
 *
 * \param [in,out] input_output Pointer to the input array where the normalization will be performed in place.
 * \param [in] size The number of elements in the input array.
 * \param [in] min_range The minimum value of the desired output range.
 * \param [in] max_range The maximum value of the desired output range.
 * \param [in] stream The CUDA stream to use for the kernel launch and memory operations.
 *
 * \note The function uses the Thrust library to find the minimum and maximum values in the array.
 *       It also calls `cudaCheckError()` to check for any CUDA errors after the kernel launch.
 */
void normalize_array(float* device_array, size_t size, float min_range, float max_range, cudaStream_t stream);

/*!
 * \brief Converts a floating-point image to an 8-bit unsigned integer image.
 *
 * This function converts a floating-point image to an 8-bit unsigned integer image by scaling the pixel values
 * to the range [0, 255]. It clamps the values to the specified minimum and maximum values, normalizes them to
 * the range [0, 1], and then scales them to [0, 255]. The result is rounded to the nearest integer.
 *
 * \param [in,out] image Pointer to the input image array where the conversion will be performed in place.
 * \param [in] size The number of elements in the input image array.
 * \param [in] minVal The minimum value in the input image array.
 * \param [in] maxVal The maximum value in the input image array.
 *
 * \note The function performs the conversion in place, meaning the input image array is modified directly.
 */
void im2uint8(float* image, size_t size, float minVal = 0.0f, float maxVal = 1.0f);
