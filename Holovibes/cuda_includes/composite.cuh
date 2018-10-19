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

# include "Common.cuh"

# include "compute_descriptor.hh"

/// Computes 3 different p slices and put them in each color
void rgb(cuComplex	*input,
	float					*output,
	const uint				frame_res,
	bool					normalize,
	const ushort red,
	const ushort blue,
	const float weight_r,
	const float weight_g,
	const float weight_b);

void postcolor_normalize(float *output,
	const uint frame_res,
	const uint real_line_size,
	holovibes::units::RectFd	selection,
	const float weight_r,
	const float weight_g,
	const float weight_b);
