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

void fill_percentile_float_in_case_of_error(float* out_percent, unsigned size_percent)
{
	for (size_t i = 0; i < size_percent; i++)
	{
		out_percent[i] = i;
	}
}

/*
** \brief Sort a copy of the array and save each of the values at h_percent % of the array in h_out_percent
** i.e. h_percent = [25f, 50f, 75f] and gpu_input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] and size_percent = 3
** gives : h_out_percent = [3, 6, 8]
*/
void percentile_float(const float *gpu_input,
					  unsigned width,
					  unsigned height,
					  unsigned offset,
				 	  const float* h_percent,
				 	  float* h_out_percent,
				 	  unsigned size_percent,
				 	  const holovibes::units::RectFd& sub_zone,
				 	  bool compute_on_sub_zone)
{
	try {
		unsigned frame_res = width * height - 2 * offset;

		if (compute_on_sub_zone)
			frame_res = sub_zone.area();

		if (frame_res == 0)
			return;

		float* gpu_input_copy_;
		cudaXMalloc((void**) &gpu_input_copy_, frame_res * sizeof(float));
		thrust::device_ptr<float> gpu_input_copy(gpu_input_copy_);

		if (compute_on_sub_zone)
			frame_memcpy(gpu_input + offset, sub_zone, width, gpu_input_copy_);
		else
			thrust::copy(gpu_input + offset, gpu_input + frame_res, gpu_input_copy);

		thrust::sort(gpu_input_copy, gpu_input_copy + frame_res);

		for (unsigned i = 0; i < size_percent; ++i)
		{
			unsigned index = h_percent[i] / 100 * frame_res;
			// copy gpu_input_copy[index] in h_out_percent[i]
			thrust::copy(gpu_input_copy + index, gpu_input_copy + index + 1, h_out_percent + i);
			cudaCheckError();
		}

		cudaXFree(gpu_input_copy_);
	}
	catch (...)
	{
		LOG_ERROR("Something went wrong, you should decrease the number of images to free some GPU memory");
		fill_percentile_float_in_case_of_error(h_out_percent, size_percent);
	}
}

