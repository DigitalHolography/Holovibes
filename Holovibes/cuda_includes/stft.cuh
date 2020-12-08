/* **************************************************************************** */
/*                       ,,                     ,,  ,,                          */
/* `7MMF'  `7MMF'       `7MM       `7MMF'   `7MF'db *MM                         */
/*   MM      MM           MM         `MA     ,V      MM                         */
/*   MM      MM  ,pW"Wq.  MM  ,pW"Wq. VM:   ,V `7MM  MM,dMMb.   .gP"Ya  ,pP"Ybd */
/*   MMmmmmmmMM 6W'   `Wb MM 6W'   `Wb MM.  M'   MM  MM    `Mb ,M'   Yb 8I   `" */
/*   MM      MM 8M     M8 MM 8M     M8 `MM A'    MM  MM     M8 8M"""""" `YMMMa. */
/*   MM      MM YA.   ,A9 MM YA.   ,A9  :MM;     MM  MM.   ,M9 YM.    , L.   I8 */
/* .JMML.  .JMML.`Ybmd9'.JMML.`Ybmd9'    VF    .JMML.P^YbmdP'   `Mbmmd' M9mmmP' */
/*                                                                              */
/* **************************************************************************** */

#pragma once

#include "Common.cuh"
#include "rect.hh"
#include "enum_img_type.hh"

using holovibes::units::RectFd;

namespace holovibes
{
	class Queue;
}

/*! \brief Compute the STFT time transform from gpu_time_transformation_queue_
 * to gpu_p_acc_buffer using plan1d wich is the data and computation descriptor
 */
void stft(holovibes::Queue	*gpu_time_transformation_queue,
		cuComplex			*gpu_p_acc_buffer,
		const cufftHandle	plan1d);

void time_transformation_cuts_begin(const cuComplex		*input,
									float				*output_xz,
									float				*output_yz,
									const ushort		xmin,
									const ushort		ymin,
									const ushort		xmax,
									const ushort		ymax,
									const ushort		width,
									const ushort		height,
									const ushort		time_transformation_size,
									const uint			acc_level_xz,
									const uint			acc_level_yz,
									const holovibes::ImgType img_type,
									cudaStream_t		stream = 0);
