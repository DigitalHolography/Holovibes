/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#pragma once

#include "common.cuh"

void filter2D(cuComplex* input,
              const float* mask,
              const uint batch_size,
              const cufftHandle plan2d,
              const camera::FrameDescriptor& desc,
              const cudaStream_t stream);

void gen_filter2d_squares_mask(float *in_out,
                               const uint width,
                               const uint height,
                               const uint sq_in_radius,
                               const uint sq_out_radius,
                               const cudaStream_t stream);