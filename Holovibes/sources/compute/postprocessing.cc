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

namespace holovibes
{
	namespace compute
	{
		Postprocessing::Postprocessing(FnVector& fn_vect,
			const CoreBuffers& buffers,
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
				/* kst_size */
				int size = static_cast<int>(cd_.convo_matrix.size());
				/* Build the kst 3x3 matrix */
				std::unique_ptr<float[]> kst_complex_cpu = std::make_unique<float[]>(size);
				for (int i = 0; i < size; ++i)
					kst_complex_cpu[i] = cd_.convo_matrix[i];

				/* gpu_kernel_buffer */
				gpu_kernel_buffer_.resize(size);
				cudaMemcpy(gpu_kernel_buffer_, kst_complex_cpu.get(), sizeof(float) * size, cudaMemcpyHostToDevice);
			}
			if (cd_.flowgraphy_enabled || cd_.convolution_enabled)
				gpu_special_queue_.resize(fd_.frame_res() * cd_.special_buffer_size);
		}

		void Postprocessing::insert_vibrometry()
		{
			if (cd_.vibrometry_enabled)
			{
				cufftComplex* qframe = buffers_.gpu_input_buffer_ + fd_.frame_res();
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
				gpu_special_queue_start_index_ = 0;
				gpu_special_queue_max_index_ = cd_.special_buffer_size;
				fn_vect_.push_back([=]() {
					convolution_kernel(
						buffers_.gpu_input_buffer_,
						gpu_special_queue_,
						fd_.frame_res(),
						fd_.width,
						gpu_kernel_buffer_,
						cd_.convo_matrix_width,
						cd_.convo_matrix_height,
						cd_.convo_matrix_z,
						gpu_special_queue_start_index_,
						gpu_special_queue_max_index_);
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
						buffers_.gpu_input_buffer_,
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