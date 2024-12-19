/*! \file
 *
 * \brief declaration of stft compute function
 */
#pragma once

#include "common.cuh"
#include "rect.hh"
#include "enum_img_type.hh"

using holovibes::units::RectFd;

namespace holovibes
{
class Queue;
} // namespace holovibes

/*! \brief Compute the STFT time transform from gpu_time_transformation_queue_
 * to gpu_p_acc_buffer using plan1d wich is the data and computation descriptor
 */
void stft(cuComplex* output, cuComplex* input, const cufftHandle plan1d);

void time_transformation_cuts_begin(float* output_xz,
                                    float* output_yz,
                                    const cuComplex* input,
                                    const ushort xmin,
                                    const ushort ymin,
                                    const ushort xmax,
                                    const ushort ymax,
                                    const ushort width,
                                    const ushort height,
                                    const ushort time_transformation_size,
                                    const uint acc_level_xz,
                                    const uint acc_level_yz,
                                    const holovibes::ImgType img_type,
                                    const cudaStream_t stream);
