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

#include "rect.hh"
#include <qrect.h>

/// Extract the part of *input described by frame
void extract_frame(const float	*input,
				float			*output,
				const uint		input_w,
				const holovibes::units::RectFd&	frame);

/// Resize the image
void gpu_resize(const float		*input,
				float			*output,
				QPoint			old_size,
				QPoint			new_size,
				cudaStream_t	stream = 0);

/// Mirrors the image inplace on both axis
void rotation_180(float				*frame,
					QPoint			size,
					cudaStream_t	stream = 0);

