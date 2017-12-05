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
#include "compute_bundles.hh"
#include "compute_bundles_2d.hh"
#include "tools_conversion.cuh"
#include "composite.cuh"

namespace holovibes
{
	namespace compute
	{
		Converts::Converts(FnVector& fn_vect,
			const CoreBuffers& buffers,
			const Stft_env& stft_env,
			const cufftHandle& plan2d,
			ComputeDescriptor& cd,
			const camera::FrameDescriptor& input_fd,
			const camera::FrameDescriptor& output_fd)
			: fn_vect_(fn_vect)
			, buffers_(buffers)
			, stft_env_(stft_env)
			, unwrap_res_()
			, unwrap_res_2d_()
			, plan2d_(plan2d)
			, cd_(cd)
			, fd_(input_fd)
			, output_fd_(output_fd)
		{}

		void Converts::insert_to_float(bool unwrap_2d_requested)
		{
			if (cd_.img_type == Composite)
				insert_to_composite();
			else if (cd_.img_type == Modulus)
				insert_to_modulus();
			else if (cd_.img_type == SquaredModulus)
				insert_to_squaredmodulus();
			if (cd_.img_type == Argument)
				insert_to_argument(unwrap_2d_requested);
			else if (cd_.img_type == PhaseIncrease)
				insert_to_phase_increase(unwrap_2d_requested);
			else if (cd_.img_type == Complex)
				insert_to_complex();
		}

		void Converts::insert_to_ushort()
		{
			insert_main_ushort();
			if (cd_.stft_view_enabled)
				insert_slice_ushort();
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
				composite(stft_env_.gpu_stft_buffer_.get(),
					buffers_.gpu_float_buffer_,
					fd_.frame_res(),
					fd_.width,
					cd_.composite_auto_weights_,
					cd_.getCompositeZone(),
					cd_.component_r,
					cd_.component_g,
					cd_.component_b);
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

		void Converts::insert_to_argument(bool unwrap_2d_requested)
		{
			fn_vect_.push_back([=]() {
				complex_to_argument(buffers_.gpu_input_buffer_, buffers_.gpu_float_buffer_, fd_.frame_res()); });

			if (unwrap_2d_requested)
			{
				try
				{
					if (!unwrap_res_2d_)
						unwrap_res_2d_.reset(new UnwrappingResources_2d(fd_.frame_res()));
					if (unwrap_res_2d_->image_resolution_ != fd_.frame_res())
						unwrap_res_2d_->reallocate(fd_.frame_res());

					fn_vect_.push_back([=]() {
						unwrap_2d(
							buffers_.gpu_float_buffer_,
							plan2d_,
							unwrap_res_2d_.get(),
							fd_,
							unwrap_res_2d_->gpu_angle_);
					});

					// Converting angle information in floating-point representation.
					fn_vect_.push_back([=]() {
						rescale_float_unwrap2d(
							unwrap_res_2d_->gpu_angle_,
							buffers_.gpu_float_buffer_,
							unwrap_res_2d_->minmax_buffer_,
							fd_.frame_res());
					});
				}
				catch (std::exception& e)
				{
					std::cerr << "Error while trying to convert to float in Argument :" << e.what() << std::endl;
				}
			}
		}

		void Converts::insert_to_phase_increase(bool unwrap_2d_requested)
		{
			try
			{
				if (!unwrap_res_)
					unwrap_res_.reset(new UnwrappingResources(cd_.unwrap_history_size, fd_.frame_res()));
				unwrap_res_->reset(cd_.unwrap_history_size);
				unwrap_res_->reallocate(fd_.frame_res());
				fn_vect_.push_back([=]() {
					phase_increase(
						buffers_.gpu_input_buffer_,
						unwrap_res_.get(),
						fd_.frame_res());
				});

				if (unwrap_2d_requested)
				{
					if (!unwrap_res_2d_)
						unwrap_res_2d_.reset(new UnwrappingResources_2d(fd_.frame_res()));

					if (unwrap_res_2d_->image_resolution_ != fd_.frame_res())
						unwrap_res_2d_->reallocate(fd_.frame_res());

					fn_vect_.push_back([=]() {
						unwrap_2d(
							unwrap_res_->gpu_angle_current_,
							plan2d_,
							unwrap_res_2d_.get(),
							fd_,
							unwrap_res_2d_->gpu_angle_);
					});

					// Converting angle information in floating-point representation.
					fn_vect_.push_back([=]() {
						rescale_float_unwrap2d(
							unwrap_res_2d_->gpu_angle_,
							buffers_.gpu_float_buffer_,
							unwrap_res_2d_->minmax_buffer_,
							fd_.frame_res());
					});
				}
				else
					fn_vect_.push_back([=]() {
					rescale_float(
						unwrap_res_->gpu_angle_current_,
						buffers_.gpu_float_buffer_,
						fd_.frame_res()); });
			}
			catch (std::exception& e)
			{
				std::cerr << "Error while trying to convert to float in Phase increase :" << e.what() << std::endl;
			}
		}

		void Converts::insert_main_ushort()
		{
			fn_vect_.push_back([=]() {
				float_to_ushort(
					buffers_.gpu_float_buffer_,
					buffers_.gpu_output_buffer_,
					buffers_.gpu_float_buffer_size_,
					output_fd_.depth);
			});
		}

		void Converts::insert_slice_ushort()
		{
			fn_vect_.push_back([=]() {
				float_to_ushort(
					static_cast<float *>(buffers_.gpu_float_cut_xz_.get()),
					buffers_.gpu_ushort_cut_xz_,
					stft_env_.gpu_stft_slice_queue_xz->get_frame_desc().frame_res(),
					2.f);
			});
			fn_vect_.push_back([=]() {
				float_to_ushort(
					static_cast<float *>(buffers_.gpu_float_cut_yz_.get()),
					buffers_.gpu_ushort_cut_yz_,
					stft_env_.gpu_stft_slice_queue_yz->get_frame_desc().frame_res(),
					2.f);
			});
		}
	}
}
