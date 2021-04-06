/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#include "unpack_12_to_16bit.cuh"

__global__ void kernel_unpack_12_to_16bit(short* output,
                                          const size_t output_size,
                                          const unsigned char* input,
                                          const size_t input_size)
{
    const uint index = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const uint index_12bit = (index * 3) / 2;

    // In ROI
    if (index < output_size && index_12bit < input_size)
    {
        output[index] =
            (input[index_12bit] << 8) | ((input[index_12bit + 1] & 0xF0));
        output[index + 1] = ((input[index_12bit + 1] & 0x0F) << 12) |
                            (input[index_12bit + 2] << 4);
    }
}

// input and output do not overlap !!
void unpack_12_to_16bit(short* output,
                        const size_t output_size,
                        const unsigned char* input,
                        const size_t input_size,
                        const cudaStream_t stream)
{
    uint threads = THREADS_128;
    uint blocks = map_blocks_to_problem(output_size / 2, threads);

    kernel_unpack_12_to_16bit<<<blocks, threads, 0, stream>>>(output,
                                                              output_size,
                                                              input,
                                                              input_size);
    cudaCheckError();
}
