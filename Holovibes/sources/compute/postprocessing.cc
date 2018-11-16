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
#include "vibrometry.cuh"
#include "convolution.cuh"
#include "flowgraphy.cuh"
#include "tools.cuh"
#include "tools_compute.cuh"
using holovibes::cuda_tools::CufftHandle;

namespace holovibes
{
	namespace compute
	{
		Postprocessing::Postprocessing(FnVector& fn_vect,
			CoreBuffers& buffers,
			const camera::FrameDescriptor& input_fd,
			ComputeDescriptor& cd)
			: gpu_special_queue_()
			, gpu_kernel_buffer_()
			, gpu_special_queue_start_index_(0)
			, gpu_special_queue_max_index_(0)
			, fn_vect_(fn_vect)
			, buffers_(buffers)
			, fd_(input_fd)
			, cd_(cd)
		{	
			allocate_buffers();
		}

		void Postprocessing::allocate_buffers()
		{
			if (cd_.convolution_enabled)
			{
				int size = cd_.convo_matrix.size();

				gpu_kernel_buffer_.resize(size * sizeof(float));
				cudaMemcpy(gpu_kernel_buffer_, cd_.convo_matrix.data(), sizeof(float) * size, cudaMemcpyHostToDevice);
	
			}
			if (cd_.flowgraphy_enabled || cd_.convolution_enabled)
			{
				buffers_.gpu_convolution_buffer_.resize(fd_.frame_res() * sizeof(float));
			}
		}

		void Postprocessing::insert_vibrometry()
		{
			if (cd_.vibrometry_enabled)
			{
				cufftComplex* qframe = buffers_.gpu_input_buffer_.get() + fd_.frame_res();
				fn_vect_.push_back([=]() {
					frame_ratio(
						buffers_.gpu_input_buffer_,
						qframe,
						buffers_.gpu_input_buffer_,
						fd_.frame_res());
				});
			}
		}

		void Postprocessing::insert_convolution()
		{
			if (cd_.convolution_enabled)
			{
				fn_vect_.push_back([=]() {
					convolution_kernel(
						buffers_.gpu_float_buffer_,
						buffers_.gpu_convolution_buffer_,
						fd_.width,
						fd_.height,
						gpu_kernel_buffer_);
				});
			}
		}

		void Postprocessing::insert_flowgraphy()
		{
			if (cd_.flowgraphy_enabled)
			{
				gpu_special_queue_start_index_ = 0;
				gpu_special_queue_max_index_ = cd_.special_buffer_size;
				fn_vect_.push_back([=]() {
					convolution_flowgraphy(
						buffers_.gpu_input_buffer_,  //want gpu_float_buffer_ (same file)
						gpu_special_queue_,
						gpu_special_queue_start_index_,
						gpu_special_queue_max_index_,
						fd_.frame_res(),
						fd_.width,
						cd_.flowgraphy_level);
				});
			}
		}
	}
}