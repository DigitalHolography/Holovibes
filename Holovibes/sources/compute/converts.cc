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

#include "converts.hh"
#include "frame_desc.hh"
#include "pipeline_utils.hh"
#include "compute_descriptor.hh"
#include "icompute.hh"
#include "tools_conversion.cuh"
#include "composite.cuh"

namespace holovibes
{
	namespace compute
	{
		Converts::Converts(FnVector& fn_vect,
			const CoreBuffers& buffers,
			cufftComplex* const& gpu_stft_buffer,
			const std::unique_ptr<Queue>& gpu_3d_vision,
			ComputeDescriptor& cd,
			const camera::FrameDescriptor& input_fd)
			: fn_vect_(fn_vect)
			, buffers_(buffers)
			, gpu_stft_buffer_(gpu_stft_buffer)
			, gpu_3d_vision_(gpu_3d_vision)
			, cd_(cd)
			, fd_(input_fd)
		{}

		void Converts::insert_to_float()
		{
			if (cd_.img_type == Composite)
				insert_to_composite();
			else if (cd_.img_type == Modulus)
			{
				if (cd_.vision_3d_enabled)
					insert_to_modulus_vision3d();
				else
					insert_to_modulus();
			}
			else if (cd_.img_type == SquaredModulus)
				insert_to_squaredmodulus();
			/*else if (cd_.img_type == Argument)
			{
				fn_vect_.push_back(std::bind(
					complex_to_argument,
					gpu_input_frame_ptr_,
					gpu_float_buffer_,
					input_fd.frame_res(),
					static_cast<cudaStream_t>(0)));

				if (unwrap_2d_requested_.load())
				{
					try
					{
						if (!unwrap_res_2d_)
							unwrap_res_2d_.reset(new UnwrappingResources_2d(input_.get_pixels()));
						if (unwrap_res_2d_->image_resolution_ != input_.get_pixels())
							unwrap_res_2d_->reallocate(input_.get_pixels());

						fn_vect_.push_back(std::bind(
							unwrap_2d,
							gpu_float_buffer_,
							plan2d_,
							unwrap_res_2d_.get(),
							input_.get_frame_desc(),
							unwrap_res_2d_->gpu_angle_,
							static_cast<cudaStream_t>(0)));

						// Converting angle information in floating-point representation.
						fn_vect_.push_back(std::bind(
							rescale_float_unwrap2d,
							unwrap_res_2d_->gpu_angle_,
							gpu_float_buffer_,
							unwrap_res_2d_->minmax_buffer_,
							input_fd.frame_res(),
							static_cast<cudaStream_t>(0)));
					}
					catch (std::exception& e)
					{
						std::cout << e.what() << std::endl;
					}
				}
				else
				{
					// Converting angle information in floating-point representation.
					fn_vect_.push_back(std::bind(
						rescale_argument,
						gpu_float_buffer_,
						input_fd.frame_res(),
						static_cast<cudaStream_t>(0)));
				}
			}*/
			else if (cd_.img_type == Complex)
				insert_to_complex();
		}

		void Converts::insert_to_modulus()
		{
			fn_vect_.push_back([=]() {
				complex_to_modulus(
					buffers_.gpu_input_buffer_,
					buffers_.gpu_float_buffer_,
					fd_.frame_res());
			});
		}

		void Converts::insert_to_squaredmodulus()
		{
			fn_vect_.push_back([=]() {
				complex_to_squared_modulus(
					buffers_.gpu_input_buffer_,
					buffers_.gpu_float_buffer_,
					fd_.frame_res());
			});
		}

		void Converts::insert_to_composite()
		{
			fn_vect_.push_back([=]() {
				Component *comps[] = { &cd_.component_r, &cd_.component_g, &cd_.component_b };
				for (Component* component : comps)
					if (component->p_max < component->p_min || component->p_max >= cd_.nsamples)
						return;
				composite(gpu_stft_buffer_,
					buffers_.gpu_float_buffer_,
					fd_.frame_res(),
					cd_.composite_auto_weights_,
					cd_.component_r.p_min,
					cd_.component_r.p_max,
					cd_.component_r.weight,
					cd_.component_g.p_min,
					cd_.component_g.p_max,
					cd_.component_g.weight,
					cd_.component_b.p_min,
					cd_.component_b.p_max,
					cd_.component_b.weight);
			});
		}

		void Converts::insert_to_complex()
		{
			fn_vect_.push_back([=]() {
				cudaMemcpy(
					buffers_.gpu_output_buffer_,
					buffers_.gpu_input_buffer_,
					fd_.frame_res() << 3,
					cudaMemcpyDeviceToDevice);
			});
		}

		void Converts::insert_to_modulus_vision3d()
		{
			fn_vect_.push_back([=]() {
				complex_to_modulus(
					gpu_stft_buffer_,
					static_cast<float *>(gpu_3d_vision_->get_buffer()),
					fd_.frame_res() * cd_.nsamples);
			});
		}
	}
}
