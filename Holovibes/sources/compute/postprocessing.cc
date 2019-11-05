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

#include "postprocessing.hh"
#include "icompute.hh"
#include "compute_descriptor.hh"
#include "convolution.cuh"
#include "tools.cuh"
#include "tools_compute.cuh"
#include "contrast_correction.cuh"
#include "hsv.cuh"
using holovibes::cuda_tools::CufftHandle;

namespace holovibes
{
	namespace compute
	{
		Postprocessing::Postprocessing(FnVector& fn_vect,
			CoreBuffers& buffers,
			const camera::FrameDescriptor& input_fd,
			ComputeDescriptor& cd)
			: gpu_kernel_buffer_()
			, fn_vect_(fn_vect)
			, buffers_(buffers)
			, fd_(input_fd)
			, cd_(cd)
			, plan_(input_fd.width, input_fd.height, CUFFT_C2C)
		{	
			allocate_buffers();
		}

		void Postprocessing::allocate_buffers()
		{
			if (cd_.convolution_enabled)
			{
				size_t size = cd_.convo_matrix.size();

				gpu_kernel_buffer_.resize(size * sizeof(float));
				cudaMemcpy(gpu_kernel_buffer_, cd_.convo_matrix.data(), sizeof(float) * size, cudaMemcpyHostToDevice);
				shift_corners(gpu_kernel_buffer_, cd_.convo_matrix_width, cd_.convo_matrix_height);
				
				buffers_.gpu_convolution_buffer_.resize(fd_.frame_res() * sizeof(float));
			}
		}

		void Postprocessing::insert_convolution_composite()
		{
			float *tmp_hsv_arr;
			cudaMalloc(&tmp_hsv_arr, sizeof(float) * fd_.frame_res() * 3);
			cudaCheckError();

			from_interweaved_components_to_distinct_components(buffers_.gpu_float_buffer_, tmp_hsv_arr, fd_.frame_res());

			convolution_kernel(tmp_hsv_arr, buffers_.gpu_convolution_buffer_, &plan_, fd_.width, fd_.height, 
				gpu_kernel_buffer_, cd_.divide_convolution_enabled, true);
			convolution_kernel(tmp_hsv_arr + fd_.frame_res() , buffers_.gpu_convolution_buffer_, &plan_, fd_.width, fd_.height, 
				gpu_kernel_buffer_, cd_.divide_convolution_enabled, true);
			convolution_kernel(tmp_hsv_arr + (fd_.frame_res() * 2), buffers_.gpu_convolution_buffer_, &plan_, fd_.width, fd_.height, 
				gpu_kernel_buffer_, cd_.divide_convolution_enabled, true);

			from_distinct_components_to_interweaved_components(tmp_hsv_arr, buffers_.gpu_float_buffer_, fd_.frame_res());

			cudaFree(tmp_hsv_arr);
		}

		void Postprocessing::insert_convolution()
		{
			if (!cd_.convolution_enabled)
				return;
			
			if (cd_.img_type != ImgType::Composite)
			{
				fn_vect_.push_back([=]() {
					convolution_kernel(
						buffers_.gpu_float_buffer_,
						buffers_.gpu_convolution_buffer_,
						&plan_,
						fd_.width,
						fd_.height,
						gpu_kernel_buffer_,
						cd_.divide_convolution_enabled,
						true);
				});
			}
			else
			{
				fn_vect_.push_back([=]() {
					insert_convolution_composite();
				});
			}
		}
	}

	/*! Returns the value of a gaussian filter at coordinates (x, y) */
	static float get_gaussian_value(int x, int y)
	{
		// G(x, y) = (1 / (2 * pi * sigma²)) * exp(-(x² + y²) / (2 * sigma²))
		constexpr float sigma = 5.0f;
		constexpr float sigma_sq = sigma * sigma;
		constexpr float pi = M_PI;

		constexpr float	left = 1.0f / (2.0f * pi * sigma_sq);
		float right = std::expf(-(x * x + y * y) / (2.0f * sigma_sq));

		return left * right;
	}

	std::vector<float> compute_gaussian_kernel(int width, int height)
	{
		std::vector<float> kernel(width * height, 0.0f);

		int width_2 = width / 2;
		int height_2 = height / 2;

		for (int y = 0; y < height; ++y)
		{
			for (int x = 0; x < width; ++x)
			{
				unsigned matrix_idx = y * width + x;
				int x_idx = x - width_2;
				int y_idx = y - height_2;
				kernel[matrix_idx] = get_gaussian_value(x_idx, y_idx);
			}
		}

		return kernel;
	}
}