#include "stft.cuh"
#include "common.cuh"

#include <cassert>

using holovibes::ImageTypeEnum;

// Short-Time Fourier Transform
void stft(cuComplex* input, cuComplex* output, const cufftHandle plan1d)
{
    // FFT 1D
    cufftSafeCall(cufftExecC2C(plan1d, input, output, CUFFT_FORWARD));

    // No sync needed since all the kernels are executed on stream compute
}

__global__ static void fill_32bit_slices(const cuComplex* input,
                                         float* output_xz,
                                         float* output_yz,
                                         const uint xmin,
                                         const uint ymin,
                                         const uint xmax,
                                         const uint ymax,
                                         const uint frame_size,
                                         const uint output_size,
                                         const uint width,
                                         const uint height,
                                         const uint acc_level_xz,
                                         const uint acc_level_yz,
                                         const holovibes::ImageTypeEnum img_type,
                                         const uint time_transformation_size)
{
    const uint id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < height * time_transformation_size)
    {
        float sum = 0;
        for (int x = xmin; x <= xmax; ++x)
        {
            float pixel_float = 0;
            cuComplex pixel =
                input[x + (id / time_transformation_size) * width + (id % time_transformation_size) * frame_size];
            if (img_type == ImageTypeEnum::Modulus || img_type == ImageTypeEnum::PhaseIncrease || img_type == ImageTypeEnum::Composite)
                pixel_float = hypotf(pixel.x, pixel.y);
            else if (img_type == ImageTypeEnum::SquaredModulus)
            {
                pixel_float = hypotf(pixel.x, pixel.y);
                pixel_float *= pixel_float;
            }
            else if (img_type == ImageTypeEnum::Argument)
                pixel_float = (atanf(pixel.y / pixel.x) + M_PI_2);
            sum += pixel_float;
        }
        output_yz[id] = sum / static_cast<float>(xmax - xmin + 1);
    }
    /* ********** */
    if (id < width * time_transformation_size)
    {
        float sum = 0;
        for (int y = ymin; y <= ymax; ++y)
        {
            float pixel_float = 0;
            cuComplex pixel = input[(y * width) + (id / width) * frame_size + id % width];
            if (img_type == ImageTypeEnum::Modulus || img_type == ImageTypeEnum::PhaseIncrease || img_type == ImageTypeEnum::Composite)
                pixel_float = hypotf(pixel.x, pixel.y);
            else if (img_type == ImageTypeEnum::SquaredModulus)
            {
                pixel_float = hypotf(pixel.x, pixel.y);
                pixel_float *= pixel_float;
            }
            else if (img_type == ImageTypeEnum::Argument)
                pixel_float = (atanf(pixel.y / pixel.x) + M_PI_2);
            sum += pixel_float;
        }
        output_xz[id] = sum / static_cast<float>(ymax - ymin + 1);
    }
}

void time_transformation_cuts_begin(const cuComplex* input,
                                    float* output_xz,
                                    float* output_yz,
                                    const ushort xmin,
                                    const ushort ymin,
                                    const ushort xmax,
                                    const ushort ymax,
                                    const ushort width,
                                    const ushort height,
                                    const ushort time_transformation_size,
                                    const uint acc_level_xz,
                                    const uint acc_level_yz,
                                    const holovibes::ImageTypeEnum img_type,
                                    const cudaStream_t stream)
{
    const uint frame_size = width * height;
    const uint output_size = std::max(width, height) * time_transformation_size;
    const uint threads = get_max_threads_1d();
    const uint blocks = map_blocks_to_problem(output_size, threads);

    fill_32bit_slices<<<blocks, threads, 0, stream>>>(input,
                                                      output_xz,
                                                      output_yz,
                                                      xmin,
                                                      ymin,
                                                      xmax,
                                                      ymax,
                                                      frame_size,
                                                      output_size,
                                                      width,
                                                      height,
                                                      acc_level_xz,
                                                      acc_level_yz,
                                                      img_type,
                                                      time_transformation_size);

    cudaCheckError();
}
