#include "cuda_memory.cuh"
#include "map.cuh"

#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

/*!
 * \brief Adds the elements of an input frame to a sum array.
 *
 * This CUDA kernel function adds the elements of an input frame to a sum array. The operation is performed in place,
 * meaning the sum array is modified directly. Each thread in the kernel adds the corresponding element from the input
 * frame to the sum array.
 *
 * \param [in,out] input_output Pointer to the sum array where the addition will be performed in place.
 * \param [in] input Pointer to the input frame containing the elements to be added.
 * \param [in] frame_size The number of elements in the input frame and the sum array.
 *
 * \note The function performs the addition only for the elements within the specified frame size.
 */
static __global__ void
kernel_add_frame_to_sum(float* const input_output, const float* const input, const size_t frame_size)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < frame_size)
        input_output[index] += input[index];
}

void add_frame_to_sum(float* const input_output, const float* const input, const size_t size, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    kernel_add_frame_to_sum<<<blocks, threads, 0, stream>>>(input_output, input, size);
    cudaCheckError();
}

/*!
 * \brief Subtracts the elements of an input frame from a sum array.
 *
 * This CUDA kernel function subtracts the elements of an input frame from a sum array. The operation is performed in
 * place, meaning the sum array is modified directly. Each thread in the kernel subtracts the corresponding element from
 * the input frame from the sum array.
 *
 * \param [in,out] input_output Pointer to the sum array where the subtraction will be performed in place.
 * \param [in] input Pointer to the input frame containing the elements to be subtracted.
 * \param [in] frame_size The number of elements in the input frame and the sum array.
 *
 * \note The function performs the subtraction only for the elements within the specified frame size.
 */
static __global__ void
kernel_subtract_frame_from_sum(float* const input_output, const float* const input, const size_t frame_size)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < frame_size)
        input_output[index] -= input[index];
}

void subtract_frame_from_sum(float* const input_output,
                             const float* const input,
                             const size_t size,
                             cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    kernel_subtract_frame_from_sum<<<blocks, threads, 0, stream>>>(input_output, input, size);
    cudaCheckError();
}

void compute_mean(float* const output,
                  const float* const input,
                  const size_t time_window,
                  const size_t frame_size,
                  cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(frame_size, threads);
    map_divide(output, input, frame_size, time_window, stream);
}

/*!
 * \brief Centers the frames of a video by subtracting the mean frame.
 *
 * This CUDA kernel function centers the frames of a video by subtracting the mean frame from each frame.
 * The operation is performed element-wise, and the result is stored in the output array.
 *
 * \param [out] output Pointer to the output array where the centered frames will be stored.
 * \param [in] m0_video Pointer to the input video array containing the frames to be centered.
 * \param [in] m0_mean Pointer to the mean frame array.
 * \param [in] frame_size The number of elements in each frame.
 * \param [in] length_video The number of frames in the video.
 *
 * \note The function performs the centering operation only for the elements within the specified frame size and video
 * length. The modulo operation inside the kernel is used to wrap around the mean frame array, which might be
 * unoptimized.
 */
__global__ void kernel_image_centering(
    float* output, const float* m0_video, const float* m0_mean, const uint frame_size, const uint length_video)
{
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < frame_size * length_video)
        output[index] = m0_video[index] - m0_mean[index % frame_size]; // Modulo inside kernel is probably unoptimized
}

void image_centering(float* output,
                     const float* m0_video,
                     const float* m0_mean,
                     const size_t frame_size,
                     const size_t length_video,
                     const cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(frame_size * length_video, threads);

    kernel_image_centering<<<blocks, threads, 0, stream>>>(output, m0_video, m0_mean, frame_size, length_video);
    cudaCheckError();
}