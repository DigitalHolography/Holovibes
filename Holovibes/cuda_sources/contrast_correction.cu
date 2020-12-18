/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#include <numeric>
#include "map.cuh"
#include "common.cuh"

void apply_contrast_correction(float* const input,
                               const uint size,
                               const ushort dynamic_range,
                               const float min,
                               const float max,
                               const cudaStream_t stream)
{
    const float factor = dynamic_range / (max - min + FLT_EPSILON);
    const auto apply_contrast = [factor, min] __device__(float pixel) {
        return factor * (pixel - min);
    };

    map_generic(input, input, size, apply_contrast, stream);
    cudaCheckError();
}
