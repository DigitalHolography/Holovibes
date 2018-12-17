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
#include "compute_descriptor.hh"
#include "cuComplex.h"
typedef unsigned int uint;

void from_distinct_components_to_interweaved_components(const float* src, float* dst, size_t frame_res);

void from_interweaved_components_to_distinct_components(const float* src, float* dst, size_t frame_res);

void apply_percentile_and_threshold(float *gpu_arr, uint frame_res, float low_threshold, float high_threshold);


void hsv(const cuComplex *d_input,
	float *d_output,
	const uint width,
	const uint height,
	const holovibes::ComputeDescriptor &cd,
	bool is_longtimes);