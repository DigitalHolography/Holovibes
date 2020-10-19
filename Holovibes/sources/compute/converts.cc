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
#include "compute_descriptor.hh"
#include "icompute.hh"
#include "compute_bundles.hh"
#include "compute_bundles_2d.hh"
#include "tools_conversion.cuh"
#include "composite.cuh"
#include "hsv.cuh"
#include "tools_compute.cuh"
#include "logger.hh"

#include <mutex>

namespace holovibes
{
	namespace compute
	{
		Converts::Converts(FnVector& fn_vect,
			const CoreBuffers& buffers,
			const Stft_env& stft_env,
			cuda_tools::CufftHandle& plan2d,
			ComputeDescriptor& cd,
			const camera::FrameDescriptor& input_fd,
			const camera::FrameDescriptor& output_fd)
			: pmin_(0)
			, pmax_(0)
			, fn_vect_(fn_vect)
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
			insert_compute_p_accu();
			if (cd_.img_type == Composite)
				insert_to_composite();
			else if (cd_.img_type == Modulus) //img type in ui : magnitude
				insert_to_modulus();
			else if (cd_.img_type == SquaredModulus) //img type in ui : squared magnitude
				insert_to_squaredmodulus();
			else if (cd_.img_type == Argument)
				insert_to_argument(unwrap_2d_requested);
			else if (cd_.img_type == PhaseIncrease)
				insert_to_phase_increase(unwrap_2d_requested);

			if (cd_.time_filter == TimeFilter::SVD && cd_.img_type != ImgType::Composite)
			{
				fn_vect_.push_back([=]() {
					// Multiply frame by (2 ^ 16) - 1 in case of SVD
					gpu_multiply_const(buffers_.gpu_float_buffer_, fd_.frame_res(), (2 << 16) - 1);
				});
			}
		}

		void Converts::insert_to_ushort()
		{
			insert_main_ushort();
			if (cd_.stft_view_enabled)
				insert_slice_ushort();
		}

		void Converts::insert_compute_p_accu()
		{
			fn_vect_.push_back([=]() {
				pmin_ = cd_.pindex;
				if (cd_.p_accu_enabled)
					pmax_ = std::max(0, std::min(pmin_ + cd_.p_acc_level, static_cast<int>(cd_.nSize)));
				else
					pmax_ = cd_.pindex;
			});
		}

		//we use gpu_input_buffer because when nsize = 1, gpu_stft_buffer is not used.
		void Converts::insert_to_modulus()
		{
			fn_vect_.push_back([=]() {
				complex_to_modulus(
					buffers_.gpu_float_buffer_,
					stft_env_.gpu_stft_buffer_,
					pmin_,
					pmax_,
					fd_.frame_res());
			});
		}

		void Converts::insert_to_squaredmodulus()
		{
			fn_vect_.push_back([=]() {
				complex_to_squared_modulus(
					buffers_.gpu_float_buffer_,
					stft_env_.gpu_stft_buffer_,
					pmin_,
					pmax_,
					fd_.frame_res());
			});
		}

		void Converts::insert_to_composite()
		{
			fn_vect_.push_back([=]() {
				if (!is_between<ushort>(cd_.composite_p_red, 0, cd_.nSize) ||
					!is_between<ushort>(cd_.composite_p_blue, 0, cd_.nSize))
					return;

				if(cd_.composite_kind == CompositeKind::RGB)
					rgb(stft_env_.gpu_stft_buffer_.get(),
						buffers_.gpu_float_buffer_,
						fd_.frame_res(),
						cd_.composite_auto_weights_,
						cd_.composite_p_red,
						cd_.composite_p_blue,
						cd_.weight_r,
						cd_.weight_g,
						cd_.weight_b);
				else
					hsv(stft_env_.gpu_stft_buffer_.get(),
						buffers_.gpu_float_buffer_,
						fd_.width,
						fd_.height,
						cd_);

				if(cd_.composite_auto_weights_)
					postcolor_normalize(buffers_.gpu_float_buffer_,
						fd_.frame_res(),
						fd_.width, cd_.getCompositeZone(),
						cd_.weight_r,
						cd_.weight_g,
						cd_.weight_b);
			});
		}

		void Converts::insert_to_argument(bool unwrap_2d_requested)
		{
			fn_vect_.push_back([=]() {
				complex_to_argument(buffers_.gpu_float_buffer_,
					stft_env_.gpu_stft_buffer_, pmin_, pmax_, fd_.frame_res()); });

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
					LOG_ERROR(std::string("Error while trying to convert to float in Argument :") + std::string(e.what()));
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
						stft_env_.gpu_p_frame_,
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
				LOG_ERROR(std::string("Error while trying to convert to float in Phase increase :") + std::string(e.what()));
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
					buffers_.gpu_float_cut_xz_.get(),
					buffers_.gpu_ushort_cut_xz_,
					stft_env_.gpu_stft_slice_queue_xz->get_fd().frame_res(),
					2.f);
			});
			fn_vect_.push_back([=]() {
				float_to_ushort(
					buffers_.gpu_float_cut_yz_.get(),
					buffers_.gpu_ushort_cut_yz_,
					stft_env_.gpu_stft_slice_queue_yz->get_fd().frame_res(),
					2.f);
			});
		}

		void Converts::insert_complex_conversion(Queue& input)
		{
			fn_vect_.push_back([&]() {
				std::lock_guard<std::mutex> m_guard(input.getGuard());

				// Copy the data from the input queue to the input buffer
				// ALL CALL ARE ASYNCHRONOUS SINCE ALL FFTs AND MEMCPYs ARE CALLED ON STREAM 0
				input_queue_to_input_buffer(buffers_.gpu_input_buffer_.get(),
											input.get_buffer(),
											fd_.frame_res(),
											cd_.stft_steps,
											input.get_start_index(),
											input.get_max_elts(),
											fd_.depth);		

				// Reduce the size
				input.decrease_size(cd_.stft_steps);

				// Move start index
				input.increase_start_index(cd_.stft_steps);
			});
		}
	}
}
