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

#include "preprocessing.hh"
#include "icompute.hh"
#include "compute_descriptor.hh"
#include "tools_compute.cuh"
#include "interpolation.cuh"

namespace holovibes
{
	namespace compute
	{
		Preprocessing::Preprocessing(FnVector& fn_vect,
			const CoreBuffers& buffers,
			const camera::FrameDescriptor& input_fd,
			ComputeDescriptor& cd)
			: gpu_ref_diff_queue_()
			, ref_diff_state_(ENQUEUE)
			, ref_diff_counter_(0)
			, fn_vect_(fn_vect)
			, buffers_(buffers)
			, fd_(input_fd)
			, cd_(cd)
		{
		}

		void Preprocessing::allocate_ref(std::atomic<bool>& update_request)
		{
			if (update_request)
			{
				gpu_ref_diff_queue_.reset();
				ref_diff_state_ = ref_state::ENQUEUE;
				ref_diff_counter_ = cd_.ref_diff_level;

				if (cd_.ref_diff_enabled || cd_.ref_sliding_enabled)
				{
					camera::FrameDescriptor new_fd = fd_;
					new_fd.depth = 8;
					try
					{
						gpu_ref_diff_queue_.reset(new Queue(new_fd, cd_.ref_diff_level, "TakeRefQueue"));
						gpu_ref_diff_queue_->set_display(false);
					}
					catch (std::exception&)
					{
						//allocation_failed(1, CustomException("update_acc_parameter()", error_kind::fail_reference));
					}
				}
				update_request = false;
			}
		}


		float Preprocessing::compute_current_intensity(cufftComplex* buffer_ptr, size_t nb_pixels)
		{
			float res = average_operator_from_complex(buffer_ptr, nb_pixels);
			cudaStreamSynchronize(0);
			return res;
		}

		void Preprocessing::insert_frame_normalization()
		{
			fn_vect_.push_back([&]() {
				if (cd_.normalize_enabled) {
					float intensity = compute_current_intensity(buffers_.gpu_input_buffer_, fd_.width * fd_.height);
					
					//std::cout << "image division is enabled and  intensity is = " << intensity << std::endl;
					gpu_real_part_divide(buffers_.gpu_input_buffer_, fd_.width * fd_.height, intensity);
					}
			});
		}


		void Preprocessing::insert_interpolation()
		{
			if (cd_.interpolation_enabled)
				fn_vect_.push_back([=]() {
					const float ratio = cd_.interp_lambda > 0 ? cd_.lambda / cd_.interp_lambda : 1;
					tex_interpolation(buffers_.gpu_input_buffer_, fd_.width, fd_.height, ratio); });
		}

		void Preprocessing::insert_ref()
		{
			if (cd_.ref_diff_enabled)
				fn_vect_.push_back([=]() {handle_reference(); });

			// Handling ref_sliding
			if (cd_.ref_sliding_enabled)
				fn_vect_.push_back([=]() {handle_sliding_reference(); });
		}

		void Preprocessing::handle_reference()
		{
			if (ref_diff_state_ == ENQUEUE)
			{
				queue_enqueue(buffers_.gpu_input_buffer_, gpu_ref_diff_queue_.get());
				ref_diff_counter_--;
				if (ref_diff_counter_ == 0)
				{
					ref_diff_state_ = COMPUTE;
					if (cd_.ref_diff_level > 1)
						mean_images(static_cast<cufftComplex *>(gpu_ref_diff_queue_->get_buffer())
							, static_cast<cufftComplex *>(gpu_ref_diff_queue_->get_buffer()),
							cd_.ref_diff_level, fd_.frame_res());
				}
			}
			else
				substract_ref(buffers_.gpu_input_buffer_, static_cast<cufftComplex *>(gpu_ref_diff_queue_->get_buffer()),
					fd_.frame_res(), 1);
		}

		void Preprocessing::handle_sliding_reference()
		{
			queue_enqueue(buffers_.gpu_input_buffer_, gpu_ref_diff_queue_.get());
			if (ref_diff_state_ == ENQUEUE)
			{
				ref_diff_counter_--;
				if (ref_diff_counter_ == 0)
					ref_diff_state_ = COMPUTE;
			}
			else
			{
				if (cd_.ref_diff_level > 1)
					mean_images(static_cast<cufftComplex *>(gpu_ref_diff_queue_->get_buffer())
						, static_cast<cufftComplex *>(gpu_ref_diff_queue_->get_buffer()),
						cd_.ref_diff_level, fd_.frame_res());
				substract_ref(buffers_.gpu_input_buffer_, static_cast<cufftComplex *>(gpu_ref_diff_queue_->get_buffer()), fd_.frame_res(), 1);
			}
		}
	}
}