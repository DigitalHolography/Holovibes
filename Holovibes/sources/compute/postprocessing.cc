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
			if (cd_.convolution_enabled && cd_.convolution_enabled_changed)
			{
				size_t width = fd_.width;
				size_t height = fd_.height;
				size_t frame_res = width * height;

				
				//No need for memset here since it will be completely overwritten by cuComplex values
				buffers_.gpu_convolution_buffer_.resize(frame_res);

				//No need for memset here since it will be memset in the actual convolution
				cuComplex_buffer_.resize(frame_res);

				gpu_kernel_buffer_.resize(frame_res);
				cudaMemset(gpu_kernel_buffer_.get(), 0, frame_res * sizeof(cuComplex));
				cudaMemcpy2D(gpu_kernel_buffer_.get(),
							 sizeof(cuComplex),
							 cd_.convo_matrix.data(),
							 sizeof(float), sizeof(float),
							 frame_res,
							 cudaMemcpyHostToDevice);
				//We compute the FFT of the kernel, once, here instead of every time the convolution subprocess is called
				shift_corners(gpu_kernel_buffer_.get(), width, height);
				cufftExecC2C(plan_, gpu_kernel_buffer_.get(), gpu_kernel_buffer_.get(), CUFFT_FORWARD);

				cd_.convolution_enabled_changed = false;
			}
		}

		void Postprocessing::insert_convolution_composite()
		{
			float *tmp_hsv_arr;
			auto width = fd_.width;
			auto height = fd_.height;
			auto frame_res = width * height;
			cudaMalloc(&tmp_hsv_arr, sizeof(float) * frame_res * 3);
			cudaCheckError();

			from_interweaved_components_to_distinct_components(buffers_.gpu_float_buffer_,
															   tmp_hsv_arr,
															   frame_res);

			convolution_kernel(tmp_hsv_arr,
							   buffers_.gpu_convolution_buffer_.get(),
							   cuComplex_buffer_.get(),
							   &plan_,
							   width,
							   height, 
							   gpu_kernel_buffer_.get(),
							   cd_.divide_convolution_enabled,
							   true);

			convolution_kernel(tmp_hsv_arr + frame_res,
							   buffers_.gpu_convolution_buffer_.get(),
							   cuComplex_buffer_.get(),
							   &plan_,
							   width,
							   height, 
							   gpu_kernel_buffer_.get(),
							   cd_.divide_convolution_enabled,
							   true);

			convolution_kernel(tmp_hsv_arr + (frame_res * 2),
							   buffers_.gpu_convolution_buffer_.get(),
							   cuComplex_buffer_.get(),
							   &plan_,
							   width,
							   height, 
							   gpu_kernel_buffer_,
							   cd_.divide_convolution_enabled,
							   true);

			from_distinct_components_to_interweaved_components(tmp_hsv_arr,
															   buffers_.gpu_float_buffer_,
															   frame_res);

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
						buffers_.gpu_float_buffer_.get(),
						buffers_.gpu_convolution_buffer_.get(),
						cuComplex_buffer_.get(),
						&plan_,
						fd_.width,
						fd_.height,
						gpu_kernel_buffer_.get(),
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
}