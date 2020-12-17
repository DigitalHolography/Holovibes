/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#pragma once

#include "common.cuh"

#include "compute_descriptor.hh"

/// Computes 3 different p slices and put them in each color
void rgb(cuComplex* input,
         float* output,
         const uint frame_res,
         bool normalize,
         const ushort red,
         const ushort blue,
         const float weight_r,
         const float weight_g,
         const float weight_b);

void postcolor_normalize(float* output,
                         const uint frame_res,
                         const uint real_line_size,
                         holovibes::units::RectFd selection,
                         const float weight_r,
                         const float weight_g,
                         const float weight_b);
