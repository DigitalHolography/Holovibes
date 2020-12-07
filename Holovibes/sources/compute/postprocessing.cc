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
#include "cuda_memory.cuh"
#include "shift_corners.cuh"
#include "map.cuh"

using holovibes::cuda_tools::CufftHandle;

namespace holovibes
{
	namespace compute
	{
		Postprocessing::Postprocessing(FunctionVector& fn_compute_vect,
			CoreBuffersEnv& buffers,
			const camera::FrameDescriptor& input_fd,
			ComputeDescriptor& cd)
			: gpu_kernel_buffer_()
			, cuComplex_buffer_()
			, hsv_arr_()
			, reduce_result_(1) // allocate an unique double
			, fn_compute_vect_(fn_compute_vect)
			, buffers_(buffers)
			, fd_(input_fd)
			, cd_(cd)
			, convolution_plan_(input_fd.height, input_fd.width,  CUFFT_C2C)
		{}

		void Postprocessing::init()
		{
			const size_t frame_res = fd_.frame_res();

			// No need for memset here since it will be completely overwritten by cuComplex values
			buffers_.gpu_convolution_buffer.resize(frame_res);

			// No need for memset here since it will be memset in the actual convolution
			cuComplex_buffer_.resize(frame_res);

			gpu_kernel_buffer_.resize(frame_res);
			cudaXMemset(gpu_kernel_buffer_.get(), 0, frame_res * sizeof(cuComplex));
			cudaSafeCall(cudaMemcpy2D(gpu_kernel_buffer_.get(),
									  sizeof(cuComplex),
									  cd_.convo_matrix.data(),
									  sizeof(float), sizeof(float),
									  frame_res,
									  cudaMemcpyHostToDevice));

			constexpr uint batch_size = 1; // since only one frame.
			// We compute the FFT of the kernel, once, here, instead of every time the convolution subprocess is called
			shift_corners(gpu_kernel_buffer_.get(), batch_size, fd_.width, fd_.height);
			cufftSafeCall(cufftExecC2C(convolution_plan_, gpu_kernel_buffer_.get(), gpu_kernel_buffer_.get(), CUFFT_FORWARD));

			hsv_arr_.resize(frame_res * 3);
		}

		void Postprocessing::dispose()
		{
			buffers_.gpu_convolution_buffer.reset(nullptr);
			cuComplex_buffer_.reset(nullptr);
			gpu_kernel_buffer_.reset(nullptr);
			hsv_arr_.reset(nullptr);
		}

		void Postprocessing::convolution_composite()
		{
			const uint frame_res = fd_.frame_res();

			from_interweaved_components_to_distinct_components(buffers_.gpu_postprocess_frame,
															   hsv_arr_.get(),
															   frame_res);

			convolution_kernel(hsv_arr_.get(),
							   buffers_.gpu_convolution_buffer.get(),
							   cuComplex_buffer_.get(),
							   &convolution_plan_,
							   frame_res,
							   gpu_kernel_buffer_.get(),
							   cd_.divide_convolution_enabled,
							   true);

			convolution_kernel(hsv_arr_.get() + frame_res,
							   buffers_.gpu_convolution_buffer.get(),
							   cuComplex_buffer_.get(),
							   &convolution_plan_,
							   frame_res,
							   gpu_kernel_buffer_.get(),
							   cd_.divide_convolution_enabled,
							   true);

			convolution_kernel(hsv_arr_.get() + (frame_res * 2),
							   buffers_.gpu_convolution_buffer.get(),
							   cuComplex_buffer_.get(),
							   &convolution_plan_,
							   frame_res,
							   gpu_kernel_buffer_,
							   cd_.divide_convolution_enabled,
							   true);

			from_distinct_components_to_interweaved_components(hsv_arr_.get(),
															   buffers_.gpu_postprocess_frame,
															   frame_res);
		}

		void Postprocessing::insert_convolution()
		{
			if (!cd_.convolution_enabled)
				return;

			if (cd_.img_type != ImgType::Composite)
			{
				fn_compute_vect_.conditional_push_back([=]() {
					convolution_kernel(
						buffers_.gpu_postprocess_frame.get(),
						buffers_.gpu_convolution_buffer.get(),
						cuComplex_buffer_.get(),
						&convolution_plan_,
						fd_.frame_res(),
						gpu_kernel_buffer_.get(),
						cd_.divide_convolution_enabled,
						true);
				});
			}
			else
			{
				fn_compute_vect_.conditional_push_back([=]() {
					convolution_composite();
				});
			}
		}

		void Postprocessing::insert_renormalize()
		{
			if (!cd_.renorm_enabled)
				return;

			fn_compute_vect_.conditional_push_back([=]() {
				uint frame_res = fd_.frame_res();
				if (cd_.img_type == ImgType::Composite)
					frame_res *= 3;
				gpu_normalize(buffers_.gpu_postprocess_frame.get(), reduce_result_.get(), frame_res, cd_.renorm_constant);
			});
		}
	}
}