#include "common.cuh"
#include "unpack.cuh"

__global__ void
kernel_unpack_12_to_16bit(short* output, const size_t output_size, const unsigned char* input, const size_t input_size)
{
    const uint index = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const uint index_12bit = (index * 3) / 2;

    // In ROI
    if (index < output_size && index_12bit < input_size)
    {
        output[index] = (input[index_12bit] << 8) | ((input[index_12bit + 1] & 0xF0));
        output[index + 1] = ((input[index_12bit + 1] & 0x0F) << 12) | (input[index_12bit + 2] << 4);
    }
}

__global__ void
kernel_unpack_10_to_16bit(short* output, const size_t output_size, const unsigned char* input, const size_t input_size)
{
    const uint index = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const uint index_10bit = (index * 5) / 4;

    // In ROI
    if (index < output_size && index_10bit < input_size)
    {
        output[index] = (input[index_10bit] << 8) | ((input[index_10bit + 1] & 0xC0));
        output[index + 1] = ((input[index_10bit + 1] & 0x3F) << 10) | ((input[index_10bit + 2] & 0xF0) << 2);
        output[index + 2] = ((input[index_10bit + 2] & 0x0F) << 12) | ((input[index_10bit + 3] & 0xFC) << 4);
        output[index + 3] = ((input[index_10bit + 3] & 0x03) << 14) | ((input[index_10bit + 4] << 6));
    }
}

// input and output must not overlap !!
void unpack_12_to_16bit(short* output,
                        const size_t output_size,
                        const unsigned char* input,
                        const size_t input_size,
                        const cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(output_size / 2, threads);

    kernel_unpack_12_to_16bit<<<blocks, threads, 0, stream>>>(output, output_size, input, input_size);
    cudaCheckError();
}

// input and output must not overlap !!
void unpack_10_to_16bit(short* output,
                        const size_t output_size,
                        const unsigned char* input,
                        const size_t input_size,
                        const cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(output_size / 4, threads);

    kernel_unpack_10_to_16bit<<<blocks, threads, 0, stream>>>(output, output_size, input, input_size);
    cudaCheckError();
}
