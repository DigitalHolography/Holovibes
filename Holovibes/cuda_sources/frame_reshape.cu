/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#include "frame_reshape.cuh"

#include "tools.hh"
#include "common.cuh"

cudaError_t embedded_frame_cpy(const char* input,
                               const uint input_width,
                               const uint input_height,
                               char* output,
                               const uint output_width,
                               const uint output_height,
                               const uint output_startx,
                               const uint output_starty,
                               const uint elm_size,
                               cudaMemcpyKind kind,
                               const cudaStream_t stream)
{
    assert(input_width + output_startx <= output_width);
    assert(input_height + output_starty <= output_height);

    char* output_write_start =
        output + elm_size * (output_starty * output_width + output_startx);
    return cudaMemcpy2DAsync(output_write_start,
                             output_width * elm_size,
                             input,
                             input_width * elm_size,
                             input_width * elm_size,
                             input_height,
                             kind,
                             stream);
}

cudaError_t embed_into_square(const char* input,
                              const uint input_width,
                              const uint input_height,
                              char* output,
                              const uint elm_size,
                              cudaMemcpyKind kind,
                              const cudaStream_t stream)
{
    uint output_startx;
    uint output_starty;
    uint square_side_len;

    if (input_width >= input_height) // Usually the case
    {
        square_side_len = input_width;
        output_startx = 0;
        output_starty = (square_side_len - input_height) / 2;
    }
    else
    {
        square_side_len = input_height;
        output_startx = (square_side_len - input_width) / 2;
        output_starty = 0;
    }
    return embedded_frame_cpy(input,
                              input_width,
                              input_height,
                              output,
                              square_side_len,
                              square_side_len,
                              output_startx,
                              output_starty,
                              elm_size,
                              kind,
                              stream);
}

static __global__ void
kernel_batched_embed_into_square(const char* input,
                                 const uint input_width,
                                 const uint input_height,
                                 char* output,
                                 const uint output_width,
                                 const uint output_height,
                                 const uint output_startx,
                                 const uint output_starty,
                                 const uint batch_size,
                                 const uint elm_size)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint x = index % output_width;
    const uint y = index / output_width;

    if (index < output_width * output_height)
    {
        for (uint i = 0; i < batch_size; i++)
        {
            const uint batch_index =
                index + i * input_width * input_height * elm_size;

            if (x < output_startx || x >= output_startx + input_width ||
                y < output_starty || y >= output_starty + input_height)
                output[batch_index] = 0;
            else
            {
                if (output_startx ==
                    0) // Horizontal black bands (top and bottom)
                    output[batch_index] =
                        input[batch_index -
                              output_starty * input_width * elm_size];
                else // Vertical black bands (left and right)
                    output[batch_index] =
                        input[batch_index -
                              (2 * y + 1) * output_startx * elm_size];
            }
        }
    }
}

void batched_embed_into_square(const char* input,
                               const uint input_width,
                               const uint input_height,
                               char* output,
                               const uint batch_size,
                               const uint elm_size,
                               const cudaStream_t stream)
{
    uint output_startx;
    uint output_starty;
    uint square_side_len;

    if (input_width >= input_height) // Usually the case
    {
        square_side_len = input_width;
        output_startx = 0;
        output_starty = (square_side_len - input_height) / 2;
    }
    else
    {
        square_side_len = input_height;
        output_startx = (square_side_len - input_width) / 2;
        output_starty = 0;
    }

    size_t threads = get_max_threads_1d();
    size_t blocks =
        map_blocks_to_problem(square_side_len * square_side_len, threads);

    kernel_batched_embed_into_square<<<blocks, threads, 0, stream>>>
                                                         (input,
                                                          input_width,
                                                          input_height,
                                                          output,
                                                          square_side_len,
                                                          square_side_len,
                                                          output_startx,
                                                          output_starty,
                                                          batch_size,
                                                          elm_size);
    cudaCheckError();
}

cudaError_t crop_frame(const char* input,
                       const uint input_width,
                       const uint input_height,
                       const uint crop_start_x,
                       const uint crop_start_y,
                       const uint crop_width,
                       const uint crop_height,
                       char* output,
                       const uint elm_size,
                       cudaMemcpyKind kind,
                       const cudaStream_t stream)
{
    assert(crop_start_x + crop_width <= input_width);
    assert(crop_start_y + crop_height <= input_height);

    const char* crop_start =
        input + elm_size * (crop_start_y * input_width + crop_start_x);
    return cudaMemcpy2DAsync(output,
                             crop_width * elm_size,
                             crop_start,
                             input_width * elm_size,
                             crop_width * elm_size,
                             crop_height,
                             kind,
                             stream);
}

cudaError_t crop_into_square(const char* input,
                             const uint input_width,
                             const uint input_height,
                             char* output,
                             const uint elm_size,
                             cudaMemcpyKind kind,
                             const cudaStream_t stream)
{
    uint crop_start_x;
    uint crop_start_y;
    uint square_side_len;

    if (input_width >= input_height)
    {
        square_side_len = input_height;
        crop_start_x = (input_width - square_side_len) / 2;
        crop_start_y = 0;
    }
    else
    {
        square_side_len = input_width;
        crop_start_x = 0;
        crop_start_y = (input_height - square_side_len) / 2;
    }

    return crop_frame(input,
                      input_width,
                      input_height,
                      crop_start_x,
                      crop_start_y,
                      square_side_len,
                      square_side_len,
                      output,
                      elm_size,
                      kind,
                      stream);
}

static __global__ void kernel_batched_crop_into_square(const char* input,
                                                       const uint input_width,
                                                       const uint input_height,
                                                       const uint crop_start_x,
                                                       const uint crop_start_y,
                                                       const uint crop_width,
                                                       const uint crop_height,
                                                       char* output,
                                                       const uint elm_size,
                                                       const uint batch_size)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = index / crop_width;

    if (index < crop_width * crop_height)
    {
        for (uint i = 0; i < batch_size; i++)
        {
            const uint batch_index =
                index + i * input_width * input_height * elm_size;

            if (crop_start_x == 0) // Horizontal black bands (top and bottom)
                output[batch_index] =
                    input[batch_index + crop_start_y * input_width * elm_size];
            else // Vertical black bands (left and right)
                output[batch_index] =
                    input[batch_index + (2 * y + 1) * crop_start_x * elm_size];
        }
    }
}

void batched_crop_into_square(const char* input,
                              const uint input_width,
                              const uint input_height,
                              char* output,
                              const uint elm_size,
                              const uint batch_size,
                              const cudaStream_t stream)
{
    uint crop_start_x;
    uint crop_start_y;
    uint square_side_len;

    if (input_width >= input_height)
    {
        square_side_len = input_height;
        crop_start_x = (input_width - square_side_len) / 2;
        crop_start_y = 0;
    }
    else
    {
        square_side_len = input_width;
        crop_start_x = 0;
        crop_start_y = (input_height - square_side_len) / 2;
    }

    size_t threads = get_max_threads_1d();
    size_t blocks =
        map_blocks_to_problem(square_side_len * square_side_len, threads);

    kernel_batched_crop_into_square<<<blocks, threads, 0, stream>>>
                                                        (input,
                                                         input_width,
                                                         input_height,
                                                         crop_start_x,
                                                         crop_start_y,
                                                         square_side_len,
                                                         square_side_len,
                                                         output,
                                                         elm_size,
                                                         batch_size);
    cudaCheckError();
}