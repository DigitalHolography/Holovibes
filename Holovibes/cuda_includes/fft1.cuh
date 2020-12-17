/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#pragma once

#include "common.cuh"

/*! \brief Find the right threads and block to call quadratic lens
 * with and call it
 */
void fft1_lens(cuComplex* lens,
               const uint lens_side_size,
               const uint frame_height,
               const uint frame_width,
               const float lambda,
               const float z,
               const float pixel_size,
               const cudaStream_t stream = 0);

/*! \brief Apply a lens and call an fft1 on the image
 *
 * \param lens the lens that will be applied to the image
 * \param plan the first paramater of cufftExecC2C that will be called
 * on the image
 */
void fft_1(cuComplex* input,
           cuComplex* output,
           const uint batch_size,
           const cuComplex* lens,
           const cufftHandle plan2D,
           const uint frame_resolution,
           const cudaStream_t stream = 0);