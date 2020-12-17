/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#include "tools_conversion.cuh"
#include "cuda_memory.cuh"
#include "map.cuh"

using camera::FrameDescriptor;

static constexpr ushort max_ushort_value = (1 << (sizeof(ushort) * 8)) - 1;
static constexpr ushort max_ushort_value_to_float =
    static_cast<float>(max_ushort_value);

/* Kernel function wrapped by complex_to_modulus. */
static __global__ void kernel_complex_to_modulus_pacc(float* output,
                                                      const cuComplex* input,
                                                      const ushort pmin,
                                                      const ushort pmax,
                                                      const uint size)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        // We use a local variable so the global memory isn't read or written
        // everytime. Only written once at the end.
        float val = 0.0f;
        for (int i = pmin; i <= pmax; i++)
        {
            const cuComplex* current_p_frame = input + i * size;

            val += hypotf(current_p_frame[index].x, current_p_frame[index].y);
        }

        output[index] = val / (pmax - pmin + 1);
    }
}

void complex_to_modulus(float* output,
                        const cuComplex* input,
                        const ushort pmin,
                        const ushort pmax,
                        const uint size,
                        const cudaStream_t stream)
{
    const uint threads = get_max_threads_1d();
    const uint blocks = map_blocks_to_problem(size, threads);

    kernel_complex_to_modulus_pacc<<<blocks, threads, 0, stream>>>(output,
                                                                   input,
                                                                   pmin,
                                                                   pmax,
                                                                   size);
    // No sync needed since everything is run on stream 0
    cudaCheckError();
}

/* Kernel function wrapped in complex_to_squared_modulus. */
static __global__ void kernel_complex_to_squared_modulus(float* output,
                                                         const cuComplex* input,
                                                         const ushort pmin,
                                                         const ushort pmax,
                                                         const uint size)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        // We use a local variable so the global memory isn't read or written
        // everytime. Only written once at the end.
        float val = 0.0f;
        for (int i = pmin; i <= pmax; i++)
        {
            const cuComplex* current_p_frame = input + i * size;
            // square of the square root of the sum of the squares of x and y
            float tmp =
                hypotf(current_p_frame[index].x, current_p_frame[index].y);
            val += tmp * tmp;
        }
        output[index] = val / (pmax - pmin + 1);
    }
}

static __device__ cuComplex device_float_to_complex(const float input)
{
    return cuComplex{input, 0.0f};
}

template <typename OTYPE, typename ITYPE, typename FUNC>
static __global__ void
kernel_input_queue_to_input_buffer(OTYPE* output,
                                   const ITYPE* const input,
                                   FUNC convert,
                                   const uint frame_res,
                                   const int batch_size,
                                   const uint current_queue_index,
                                   const uint queue_size)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < frame_res)
    {
        uint frame_copied = 0;
        // j swip through the queue from it's start index to either the end of
        // the queue or all of the batch
        for (int j = current_queue_index;
             j < queue_size && frame_copied < batch_size;
             ++j, ++frame_copied)
            output[index + frame_copied * frame_res] =
                device_float_to_complex(convert(input[index + j * frame_res]));

        // Copy might reach end of the queue so we copy the missing frames
        for (int j = 0; frame_copied < batch_size; ++frame_copied, ++j)
            output[index + frame_copied * frame_res] =
                device_float_to_complex(convert(input[index + j * frame_res]));
    }
}

void input_queue_to_input_buffer(void* output,
                                 void* input,
                                 const uint frame_res,
                                 const int batch_size,
                                 const uint current_queue_index,
                                 const uint queue_size,
                                 const uint depth)
{
    const uint threads = get_max_threads_1d();
    const uint blocks = map_blocks_to_problem(frame_res, threads);

    /* Best way we found to pass function to kernels
     *  We can't declare the lambda outside this function for some reason
     * To pass lambda like that, we need to add the --extended-lambda  flag
     */
    static const auto convert_8_bit = [] __device__(const uchar input_pixel) {
        return static_cast<float>(input_pixel * 257);
    };
    static const auto convert_16_bit = [] __device__(const ushort input_pixel) {
        return static_cast<float>(input_pixel);
    };
    static const auto convert_32_bit = [] __device__(const float input_pixel) {
        return input_pixel;
    };

    switch (depth)
    {
    case 1:
        kernel_input_queue_to_input_buffer<cuComplex, uchar>
            <<<blocks, threads>>>(reinterpret_cast<cuComplex*>(output),
                                  reinterpret_cast<uchar*>(input),
                                  convert_8_bit,
                                  frame_res,
                                  batch_size,
                                  current_queue_index,
                                  queue_size);
        break;
    case 2:
        kernel_input_queue_to_input_buffer<cuComplex, ushort>
            <<<blocks, threads>>>(reinterpret_cast<cuComplex*>(output),
                                  reinterpret_cast<ushort*>(input),
                                  convert_16_bit,
                                  frame_res,
                                  batch_size,
                                  current_queue_index,
                                  queue_size);
        break;
    case 4:
        kernel_input_queue_to_input_buffer<cuComplex, float>
            <<<blocks, threads>>>(reinterpret_cast<cuComplex*>(output),
                                  reinterpret_cast<float*>(input),
                                  convert_32_bit,
                                  frame_res,
                                  batch_size,
                                  current_queue_index,
                                  queue_size);
        break;
    }
    // No sync needed since next call (fft1 is called on default main stream
    // (0))
    cudaCheckError();
}

void complex_to_squared_modulus(float* output,
                                const cuComplex* input,
                                const ushort pmin,
                                const ushort pmax,
                                const uint size,
                                const cudaStream_t stream)
{
    const uint threads = get_max_threads_1d();
    const uint blocks = map_blocks_to_problem(size, threads);

    kernel_complex_to_squared_modulus<<<blocks, threads, 0, stream>>>(output,
                                                                      input,
                                                                      pmin,
                                                                      pmax,
                                                                      size);
    cudaDeviceSynchronize();
    cudaCheckError();
}

/* Kernel function wrapped in complex_to_argument. */
static __global__ void kernel_complex_to_argument(float* output,
                                                  const cuComplex* input,
                                                  const ushort pmin,
                                                  const ushort pmax,
                                                  const uint size)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        // We use a local variable so the global memory isn't read or written
        // everytime. Only written once at the end.
        float val = 0.0f;
        for (int i = pmin; i <= pmax; i++)
        {
            const cuComplex* current_p_frame = input + i * size;
            // Computes the arc tangent of y / x
            // We use std::atan2 in order to obtain results in [-pi; pi].
            val +=
                std::atan2(current_p_frame[index].y, current_p_frame[index].x);
        }
        output[index] = val / (pmax - pmin + 1);
    }
}

void complex_to_argument(float* output,
                         const cuComplex* input,
                         const ushort pmin,
                         const ushort pmax,
                         const uint size,
                         const cudaStream_t stream)
{
    const uint threads = get_max_threads_1d();
    const uint blocks = map_blocks_to_problem(size, threads);

    kernel_complex_to_argument<<<blocks, threads, 0, stream>>>(output,
                                                               input,
                                                               pmin,
                                                               pmax,
                                                               size);
    cudaDeviceSynchronize();
    cudaCheckError();
}

/* Find the minimum and the maximum of a floating-point array.
 *
 * The minimum and maximum can't be computed directly, because blocks
 * cannot communicate. Hence we compute local minima and maxima and
 * put them in two arrays.
 *
 * \param Size Number of threads in a block for this kernel.
 * Also, it's the size of min and max.
 * \param min Array of Size floats, which will contain local minima.
 * \param max Array of Size floats, which will contain local maxima.
 */
template <uint Size>
static __global__ void
kernel_minmax(const float* data, const size_t size, float* min, float* max)
{
    __shared__ float local_min[Size];
    __shared__ float local_max[Size];

    const uint index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index > size)
        return;
    local_min[threadIdx.x] = data[index];
    local_max[threadIdx.x] = data[index];

    __syncthreads();

    if (threadIdx.x == 0)
    {
        /* Accumulate the results of the neighbors, computing min-max values,
         * and store them in the first element of local arrays. */
        for (auto i = 1; i < Size; ++i)
        {
            if (local_min[i] < local_min[0])
                local_min[0] = local_min[i];
            if (local_max[i] > local_max[0])
                local_max[0] = local_max[i];
        }
        min[blockIdx.x] = local_min[0];
        max[blockIdx.x] = local_max[0];
    }
}

void rescale_float(const float* input,
                   float* output,
                   const uint size,
                   const cudaStream_t stream)
{
    const uint threads = THREADS_128;
    const uint blocks = map_blocks_to_problem(size, threads);

    // TODO : See if gpu_postprocess_frame could be used directly.
    cudaXMemcpy(output, input, sizeof(float) * size, cudaMemcpyDeviceToDevice);

    // Computing minimum and maximum values, in order to rescale properly.
    float* gpu_local_min;
    float* gpu_local_max;
    const uint float_blocks = sizeof(float) * blocks;
    cudaXMalloc(&gpu_local_min, float_blocks);
    cudaXMalloc(&gpu_local_max, float_blocks);

    /* We have to hardcode the template parameter, unfortunately.
     * It must be equal to the number of threads per block. */
    kernel_minmax<128>
        <<<blocks, threads, threads << 1, stream>>>(output,
                                                    size,
                                                    gpu_local_min,
                                                    gpu_local_max);
    cudaDeviceSynchronize();
    cudaCheckError();

    float* cpu_local_min = new float[blocks];
    float* cpu_local_max = new float[blocks];
    cudaXMemcpy(cpu_local_min,
                gpu_local_min,
                float_blocks,
                cudaMemcpyDeviceToHost);
    cudaXMemcpy(cpu_local_max,
                gpu_local_max,
                float_blocks,
                cudaMemcpyDeviceToHost);

    constexpr float max_intensity = max_ushort_value_to_float;
    const float min_element =
        *(std::min_element(cpu_local_min, cpu_local_min + threads));
    const float max_element =
        *(std::max_element(cpu_local_max, cpu_local_max + threads));
    const auto lambda = [min_element, max_element, max_intensity] __device__(
                            const float in) -> float {
        return (in + fabsf(min_element)) * max_intensity /
               (fabsf(max_element) + fabsf(min_element));
    };

    map_generic<float>(output, output, size, lambda, stream);
    cudaDeviceSynchronize();
    cudaCheckError();
    delete[] cpu_local_max;
    delete[] cpu_local_min;
    cudaXFree(gpu_local_min);
    cudaXFree(gpu_local_max);
}

void rescale_float_unwrap2d(float* input,
                            float* output,
                            float* cpu_buffer,
                            uint frame_res,
                            const cudaStream_t stream)
{
    float min = 0;
    float max = 0;
    const uint threads = THREADS_128;
    const uint blocks = map_blocks_to_problem(frame_res, threads);
    uint float_frame_res = sizeof(float) * frame_res;
    cudaXMemcpy(cpu_buffer, input, float_frame_res, cudaMemcpyDeviceToHost);
    auto minmax = std::minmax_element(cpu_buffer, cpu_buffer + frame_res);
    min = *minmax.first;
    max = *minmax.second;

    const auto lambda = [min, max] __device__(const float in) -> float {
        if (min < 0.f)
            return (in + fabs(min)) / (fabs(min) + max) *
                   max_ushort_value_to_float;
        else
            return (in - min) / (max - min) * max_ushort_value_to_float;
    };
    map_generic(input, output, frame_res, lambda, stream);
}

void endianness_conversion(const ushort* input,
                           ushort* output,
                           const uint batch_size,
                           const uint size,
                           const cudaStream_t stream)
{
    static const auto lambda = [] __device__(const ushort in) -> ushort {
        return (in << 8) | (in >> 8);
    };
    map_generic(input, output, size * batch_size, lambda, stream);
}

/*
 * The input data shall be restricted first to the range [0; 2^16 - 1],
 * by forcing every negative  value to 0 and every positive one
 * greater than 2^16 - 1 to 2^16 - 1.
 * Then it is truncated to unsigned short data type.
 */
static __device__ ushort device_float_to_ushort(const float input,
                                                const uint shift = 0)
{
    if (input <= 0.0f) // Negative float
        return 0;
    // Cast in uint is needed to avoid overflow
    else if ((static_cast<uint>(input) << shift) > max_ushort_value_to_float)
        return max_ushort_value;
    else
        return static_cast<ushort>(input) << shift;
}

void complex_to_uint(const cuComplex* const input,
                     uint* const output,
                     const uint size,
                     const uint shift,
                     cudaStream_t stream = 0)
{
    const auto lambda_complex_to_ushort =
        [shift] __device__(const cuComplex in) -> uint {
        /* cuComplex needs to be casted to a uint
        ** Each part (real & imaginary) are casted from float to ushort to then
        *be assembled into a uint
        ** The real part is on the left side of the uint, imaginary is on the
        *right one
        ** Here x & y are of type uint to avoid the overflow when shifting
        */
        constexpr uint size_half_uint = sizeof(uint) * 8 / 2;
        const uint x = device_float_to_ushort(in.x);
        const uint y = device_float_to_ushort(in.y);

        return ((x << size_half_uint) | y) << shift;
    };
    map_generic(input, output, size, lambda_complex_to_ushort, stream);
}

void float_to_ushort(const float* const input,
                     ushort* const output,
                     const uint size,
                     const uint shift,
                     cudaStream_t stream)
{
    const auto lambda = [shift] __device__(const float in) -> ushort {
        return device_float_to_ushort(in, shift);
    };
    map_generic(input, output, size, lambda, stream);
}

void ushort_to_shifted_ushort(const ushort* const input,
                              ushort* const output,
                              const uint size,
                              const uint shift,
                              cudaStream_t stream = 0)
{
    const auto lambda_shift_ushort =
        [shift] __device__(const ushort in) -> ushort { return in << shift; };
    map_generic(input, output, size, lambda_shift_ushort, stream);
}

void ushort_to_uchar(const ushort* input,
                     uchar* output,
                     const uint size,
                     const cudaStream_t stream)
{
    static const auto lambda = [] __device__(const ushort in) -> uchar {
        return in >> (sizeof(uchar) * 8);
    };
    map_generic(input, output, size, lambda, stream);
}

void uchar_to_shifted_uchar(const uchar* input,
                            uchar* output,
                            const uint size,
                            const uint shift,
                            cudaStream_t stream = 0)
{
    const auto lambda_shift_uchar =
        [shift] __device__(const uchar in) -> uchar { return in << shift; };
    map_generic(static_cast<const uchar* const>(input),
                static_cast<uchar* const>(output),
                size,
                lambda_shift_uchar,
                stream);
}

__global__ void kernel_accumulate_images(const float* input,
                                         float* output,
                                         const size_t end,
                                         const size_t max_elmt,
                                         const size_t nb_elmt,
                                         const size_t nb_pixel)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    long int pos = end; // end is excluded

    if (index < nb_pixel)
    {
        float val = 0;
        for (size_t i = 0; i < nb_elmt; i++)
        {
            // get last index when pos is out of range
            // reminder: the given input is from ciruclar queue
            pos--;
            if (pos < 0)
                pos = max_elmt - 1;

            val += input[index + pos * nb_pixel];
        }
        output[index] = val / nb_elmt;
    }
}

/*! \brief Kernel function wrapped in accumulate_images, making
** the call easier
**/
void accumulate_images(const float* input,
                       float* output,
                       const size_t start,
                       const size_t max_elmt,
                       const size_t nb_elmt,
                       const size_t nb_pixel,
                       const cudaStream_t stream)
{
    const uint threads = get_max_threads_1d();
    const uint blocks = map_blocks_to_problem(nb_pixel, threads);
    kernel_accumulate_images<<<blocks, threads, 0, stream>>>(input,
                                                             output,
                                                             start,
                                                             max_elmt,
                                                             nb_elmt,
                                                             nb_pixel);
    cudaCheckError();
}

void normalize_complex(cuComplex* image,
                       const uint size,
                       const cudaStream_t stream)
{
    static const auto lambda = [] __device__(cuComplex in) -> cuComplex {
        in.x = (in.x + 1.0f) * (max_ushort_value_to_float / 2.0f);
        in.y = (in.y + 1.0f) * (max_ushort_value_to_float / 2.0f);
        return in;
    };
    map_generic(image, image, size, lambda, stream);
}

void convert_frame_for_display(const void* input,
                               void* output,
                               const uint size,
                               const uint depth,
                               const ushort shift)
{
    if (depth == 8)
    {
        // In depth 8 the output is encoded onto a uint (for the lens)
        complex_to_uint(static_cast<const cuComplex* const>(input),
                        static_cast<uint* const>(output),
                        size,
                        shift);
    }
    else if (depth == 4)
    {
        float_to_ushort(static_cast<const float* const>(input),
                        static_cast<ushort* const>(output),
                        size,
                        shift);
    }
    else if (depth == 2)
    {
        ushort_to_shifted_ushort(static_cast<const ushort* const>(input),
                                 static_cast<ushort* const>(output),
                                 size,
                                 shift);
    }
    else if (depth == 1)
    {
        uchar_to_shifted_uchar(static_cast<const uchar* const>(input),
                               static_cast<uchar* const>(output),
                               size,
                               shift);
    }
}