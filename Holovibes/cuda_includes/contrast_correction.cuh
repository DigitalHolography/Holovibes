/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#pragma once

#include "common.cuh"

/*! \brief Make the contrast of the image depending of the
 * maximum and minimum input given by the user.
 *
 * The algortihm used is a contrast stretching, the
 * values min and max can be found thanks to the previous functions
 * or can be set by the user in case of a particular use.
 * \param input The image in gpu to correct contrast.
 * \param size Size of the image in number of pixels.
 * \param dynamic_range Range of pixel values
 * \param min Minimum pixel value of the input image.
 * \param max Maximum pixel value of the input image.
 *
 */
void apply_contrast_correction(float* const input,
                               const uint size,
                               const ushort dynamic_range,
                               const float min,
                               const float max);
