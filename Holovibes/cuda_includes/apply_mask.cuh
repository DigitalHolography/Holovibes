#pragma once

/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#pragma once

#include "cuComplex.h"
#include "cuda_runtime.h"

using uint = unsigned int;
using ushort = unsigned short;

/*! \brief This function applies a mask to a number of frames
 *
 * \param input The buffer of images to modify
 * \param mask The mask to apply to 'input'
 * \param size The number of pixels in one frame of 'input'
 * \param batch_size The number of frames of 'input'
 * \param stream The CUDA stream on which to launch the operation.
 */
void apply_mask(const float* in_out,
                const float* mask,
                const uint size,
                const uint batch_size,
                const cudaStream_t stream);

void apply_mask(const cuComplex* in_out,
                const float* mask,
                const uint size,
                const uint batch_size,
                const cudaStream_t stream);

void apply_mask(const cuComplex* in_out,
                const cuComplex* mask,
                const uint size,
                const uint batch_size,
                const cudaStream_t stream);

/*! \brief This function applies a mask to a number of frames
 *
 * \param input The buffer of images to modify
 * \param mask The mask to apply to 'input' stored in 'output'
 * \param output The output buffer of the mask application
 * \param size The number of pixels in one frame of 'input'
 * \param batch_size The number of frames of 'input'
 * \param stream The CUDA stream on which to launch the operation.
 */
void apply_mask(const float* input,
                const float* mask,
                float* output,
                const uint size,
                const uint batch_size,
                const cudaStream_t stream);

void apply_mask(const cuComplex* input,
                const float* mask,
                cuComplex* output,
                const uint size,
                const uint batch_size,
                const cudaStream_t stream);

void apply_mask(const cuComplex* input,
                const cuComplex* mask,
                cuComplex* output,
                const uint size,
                const uint batch_size,
                const cudaStream_t stream);
                

__host__ __device__ cuComplex& operator*=(cuComplex& c, const float& r)
{
    c.x = c.x * r;
    c.y = c.y * r;
    return c;
}

__host__ __device__ cuComplex& operator*=(cuComplex& c1, const cuComplex& c2)
{
    float tmpx = c1.x * c2.x - c1.y * c2.y;
    float tmpy = c1.x * c2.y + c2.x * c1.y;

    c1.x = tmpx;
    c1.y = tmpy;

    return c1;
}

__host__ __device__ cuComplex operator*(const cuComplex& c, const float& r)
{
    cuComplex n;

    n.x = c.x * r;
    n.y = c.y * r;
    return n;
}

__host__ __device__ cuComplex operator*(const cuComplex& c1, const cuComplex& r)
{
    cuComplex n;

    n.x = c1.x * r.x - c1.y * r.y;
    n.y = c1.x * r.y + r.x * c1.y;
    return n;
}