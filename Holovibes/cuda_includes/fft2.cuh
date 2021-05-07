/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#pragma once

#include "common.cuh"

/*! \brief takes input complex buffer and computes a p frame that is stored
 * at output pointer. The output pointer can be another complex buffer or the
 * same as input buffer.
 */
void fft2_lens(cuComplex* lens,
               const uint lens_side_size,
               const uint frame_height,
               const uint frame_width,
               const float lambda,
               const float z,
               const float pixel_size,
               const cudaStream_t stream);

/*! \brief takes input complex buffer and computes a p frame that is stored
 * at output pointer. The output pointer can be another complex buffer or the
 * same as input buffer.
 */
void fft_2(cuComplex* input,
           cuComplex* output,
           const uint batch_size,
           const float* filter2d_mask,
           const bool filter2d_enabled,
           const cuComplex* lens,
           const cufftHandle plan2d,
           const camera::FrameDescriptor& fd,
           const cudaStream_t stream);