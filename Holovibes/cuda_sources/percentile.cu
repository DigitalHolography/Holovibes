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

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/sort.h>
#include "tools_conversion.cuh"
#include "unique_ptr.hh"
#include "tools_compute.cuh"
#include "logger.hh"
#include "cuda_memory.cuh"

void fill_percentile_float_in_case_of_error(float* const out_percent, unsigned size_percent)
{
	for (size_t i = 0; i < size_percent; i++)
	{
		out_percent[i] = i;
	}
}

thrust::device_ptr<float> allocate_thrust(const uint frame_res)
{
	float* raw_gpu_input_copy;
	cudaXMalloc(&raw_gpu_input_copy, frame_res * sizeof(float));
	return thrust::device_ptr<float>(raw_gpu_input_copy);
}

/*
** \brief Sort a copy of the array and save each of the values at h_percent % of the array in h_out_percent
** i.e. h_percent = [25f, 50f, 75f] and gpu_input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] and size_percent = 3
** gives : h_out_percent = [3, 6, 8]
*/
void compute_percentile(thrust::device_ptr<float>& thrust_gpu_input_copy,
						  const uint frame_res,
						  const float* const h_percent,
				 	  	  float* const h_out_percent,
						  const uint size_percent)
{
	thrust::sort(thrust_gpu_input_copy, thrust_gpu_input_copy + frame_res);

	for (uint i = 0; i < size_percent; ++i)
	{
		const uint index = h_percent[i] / 100 * frame_res;

		// Copy gpu_input_copy[index] in h_out_percent[i]
		thrust::copy(thrust_gpu_input_copy + index, thrust_gpu_input_copy + index + 1, h_out_percent + i);
		cudaCheckError();
	}
}

/*
** \brief Calculate frame_res according to the width, height and required offset
*
* \param factor Multiplication factor for the offset (width for xz and height for yz)
*/
uint calculate_frame_res(const uint width,
						 const uint height,
						 const uint offset,
						 const uint factor,
						 const holovibes::units::RectFd& sub_zone,
						 const bool compute_on_sub_zone)
{
	uint frame_res;

	if (compute_on_sub_zone)
		frame_res = sub_zone.area();
	else
		frame_res = width * height - 2 * offset * factor;

	assert(frame_res > 0);

	return frame_res;
}

void compute_percentile_xz_view(const float *gpu_input,
							    const uint width,
							    const uint height,
							    uint offset,
							    const float* const h_percent,
							    float* const h_out_percent,
							    const uint size_percent,
							    const holovibes::units::RectFd& sub_zone,
							    const bool compute_on_sub_zone)
{
	uint frame_res = calculate_frame_res(width, height, offset, width, sub_zone, compute_on_sub_zone);
	offset *= width;

	thrust::device_ptr<float> thrust_gpu_input_copy(nullptr);
	try {
		thrust_gpu_input_copy = allocate_thrust(frame_res);
		if (compute_on_sub_zone)
			frame_memcpy(gpu_input + offset, sub_zone, width, thrust_gpu_input_copy.get());
		else
			thrust::copy(gpu_input + offset, gpu_input + offset + frame_res, thrust_gpu_input_copy);

		compute_percentile(thrust_gpu_input_copy, frame_res, h_percent, h_out_percent, size_percent);
	}
	catch (...)
	{
		LOG_ERROR("[Thrust] Error while computing a percentile");
		fill_percentile_float_in_case_of_error(h_out_percent, size_percent);
	}
	if (thrust_gpu_input_copy.get() != nullptr)
		cudaXFree(thrust_gpu_input_copy.get());
}

void compute_percentile_xy_view(const float *gpu_input,
								const uint width,
								const uint height,
								const float* const h_percent,
								float* const h_out_percent,
								const uint size_percent,
								const holovibes::units::RectFd& sub_zone,
								const bool compute_on_sub_zone)
{
	// Computing the contrast on xy view is the same as calculating it on the xz view without any offset
	compute_percentile_xz_view(gpu_input, width, height, 0, h_percent, h_out_percent, size_percent, sub_zone, compute_on_sub_zone);
}

void compute_percentile_yz_view(const float *gpu_input,
								const uint width,
								const uint height,
								uint offset,
								const float* const h_percent,
								float* const h_out_percent,
								const uint size_percent,
								const holovibes::units::RectFd& sub_zone,
								const bool compute_on_sub_zone)
{
	uint frame_res = calculate_frame_res(width, height, offset, height, sub_zone, compute_on_sub_zone);

	thrust::device_ptr<float> thrust_gpu_input_copy(nullptr);
	try {
		thrust_gpu_input_copy = allocate_thrust(frame_res);

		// Copy sub array (skip the 2 first columns and the 2 last columns)
		cudaSafeCall(
			cudaMemcpy2D(thrust_gpu_input_copy.get(),		 	// dst
					     (width - 2 * offset) * sizeof(float),	// dpitch
					     gpu_input + offset,					// src
					     width * sizeof(float),					// spitch
					     (width - 2 * offset) * sizeof(float),	// dwidth
					     height,								// dheight
					     cudaMemcpyDeviceToDevice));			// kind

		compute_percentile(thrust_gpu_input_copy, frame_res, h_percent, h_out_percent, size_percent);
	}
	catch (...)
	{
		LOG_ERROR("[Thrust] Error while computing a percentile");
		fill_percentile_float_in_case_of_error(h_out_percent, size_percent);
	}
	if (thrust_gpu_input_copy.get() != nullptr)
		cudaXFree(thrust_gpu_input_copy.get());
}