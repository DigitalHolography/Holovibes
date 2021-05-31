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
              const uint size,
              const cudaStream_t stream);

void update_filter2d_circles_mask(float* in_out,
                                  const uint width,
                                  const uint height,
                                  const uint radius_low,
                                  const uint radius_high,
                                  const uint smooth_low,
                                  const uint smooth_high,
                                  const cudaStream_t stream);