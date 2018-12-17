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

#include <numeric>
#include "contrast_correction.cuh"
#include "tools_compute.cuh"
#include "min_max.cuh"
#include "percentile.cuh"

static __global__
void apply_contrast(float		*input,
					const uint	size,
					const float	factor,
					const float	min)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size)
	{
		input[index] = factor * (input[index] - min);
	}
}

void manual_contrast_correction(float			*input,
								const uint		size,
								const ushort	dynamic_range,
								const float		min,
								const float		max,
								cudaStream_t	stream)
{
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(size, threads);
	float test_min = min;
	float test_max = max;

	const float factor = dynamic_range / (test_max - test_min + FLT_EPSILON);
	apply_contrast << <blocks, threads, 0, stream >> > (input, size, factor, test_min);
	cudaCheckError();
}

void auto_contrast_correction(float	*input,
	const uint		size,
	const uint		offset,
	float			*min,
	float			*max,
	float			contrast_threshold_low_percentile,
	float			contrast_threshold_high_percentile)
{
	const uint threads = get_max_threads_1d();
	uint blocks = map_blocks_to_problem(size, threads);
	float percent_out[2];
	const float percent_in_h[2] =
	{
		contrast_threshold_low_percentile, contrast_threshold_high_percentile
	};

	percentile_float(input + offset, size - offset, percent_in_h, percent_out, 2);
	*min = percent_out[0];
	*max = percent_out[1];
	
	*min = ((*min < 1.0f) ? (1.0f) : (*min));
	*max = ((*max < 1.0f) ? (1.0f) : (*max));
}
