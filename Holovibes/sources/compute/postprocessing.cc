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
#include "nppi_data.hh"
#include "nppi_functions.hh"
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

		void Postprocessing::insert_renormalize()
		{
			if (!cd_.renorm_enabled)
				return;

			fn_vect_.push_back([=]() {
				cuda_tools::NppiData nppi_data(fd_.width, fd_.height);
				cuda_tools::nppi_normalize(buffers_.gpu_float_buffer_.get(), nppi_data);
			});
		}
	}
}