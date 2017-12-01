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

#include "contrast.hh"
#include "frame_desc.hh"
#include "pipeline_utils.hh"
#include "icompute.hh"
#include "compute_descriptor.hh"
#include "concurrent_deque.hh"
#include "contrast_correction.cuh"
#include "average.cuh"

namespace holovibes
{
	namespace compute
	{
		Contrast::Contrast(FnVector& fn_vect,
			const CoreBuffers& buffers,
			Average_env& average_env,
			ComputeDescriptor& cd,
			const camera::FrameDescriptor& input_fd,
			const camera::FrameDescriptor& output_fd,
			const std::unique_ptr<Queue>& gpu_3d_vision,
			ICompute* Ic)
			: fn_vect_(fn_vect)
			, buffers_(buffers)
			, average_env_(average_env)
			, cd_(cd)
			, input_fd_(input_fd)
			, fd_(output_fd)
			, gpu_3d_vision_(gpu_3d_vision)
			, Ic_(Ic)
		{
		}


		void Contrast::insert_fft_shift()
		{
			if (cd_.shift_corners_enabled)
			{
				uint size_x = buffers_.gpu_float_buffer_size_ / (sizeof(float) * fd_.height);
				fn_vect_.push_back([=]() {
					shift_corners(
						buffers_.gpu_float_buffer_,
						size_x,
						fd_.height);
				});
			}
		}

		void Contrast::insert_average(std::atomic<bool>& record_request)
		{
			//TODO: allowing both at the same time
			if (record_request)
			{
				insert_average_record();
				record_request = false;
			}
			else
				insert_main_average();
		}

		void Contrast::insert_log()
		{
			if (cd_.log_scale_slice_xy_enabled)
				insert_main_log();
			if (cd_.stft_view_enabled)
				insert_slice_log();
		}

		void Contrast::insert_contrast(std::atomic<bool>& autocontrast_request)
		{
			insert_autocontrast(autocontrast_request);
			if (cd_.contrast_enabled)
			{
				if (cd_.vision_3d_enabled)
					insert_vision3d_contrast();
				else
					insert_main_contrast();

				if (cd_.stft_view_enabled)
					insert_slice_contrast();
			}
		}

		//----------

		void Contrast::insert_main_average()
		{
			units::RectFd signalZone;
			units::RectFd noiseZone;
			cd_.signalZone(signalZone, AccessMode::Get);
			cd_.noiseZone(noiseZone, AccessMode::Get);
			fn_vect_.push_back([=]() {
				average_env_.average_output_->push_back(
					make_average_plot(
						buffers_.gpu_float_buffer_,
						input_fd_.width,
						input_fd_.height,
						signalZone,
						noiseZone));
			});
		}

		void Contrast::insert_average_record()
		{
			units::RectFd signalZone;
			units::RectFd noiseZone;
			cd_.signalZone(signalZone, AccessMode::Get);
			cd_.noiseZone(noiseZone, AccessMode::Get);
			fn_vect_.push_back([=]() {average_record_caller(
				buffers_.gpu_float_buffer_,
				input_fd_.width,
				input_fd_.height,
				signalZone,
				noiseZone);
			});
		}

		void Contrast::insert_main_log()
		{
			uint size = buffers_.gpu_float_buffer_size_ / sizeof(float);
			fn_vect_.push_back([=]() {apply_log10(buffers_.gpu_float_buffer_, size); });
		}

		void Contrast::insert_slice_log()
		{
			uint size = fd_.width * cd_.nsamples;
			if (cd_.log_scale_slice_xz_enabled)
				fn_vect_.push_back([=]() {apply_log10(buffers_.gpu_float_cut_xz_, size); });
			if (cd_.log_scale_slice_yz_enabled)
				fn_vect_.push_back([=]() {apply_log10(buffers_.gpu_float_cut_yz_, size); });
		}

		void Contrast::insert_vision3d_contrast()
		{
			float *buffer = static_cast<float *>(gpu_3d_vision_->get_buffer());
			buffer += gpu_3d_vision_->get_pixels() * cd_.pindex;
			uint size = fd_.frame_res() * (cd_.nsamples - cd_.pindex);
			fn_vect_.push_back([=]() {
				manual_contrast_correction(
					buffer,
					size,
					65535,
					cd_.contrast_min_slice_xy,
					cd_.contrast_max_slice_xy);
			});
		}

		void Contrast::insert_main_contrast()
		{
			uint size = buffers_.gpu_float_buffer_size_ / sizeof(float);
			fn_vect_.push_back([=]() {
				manual_contrast_correction(
					buffers_.gpu_float_buffer_,
					size,
					65535,
					cd_.contrast_min_slice_xy,
					cd_.contrast_max_slice_xy);
			});
		}

		void Contrast::insert_slice_contrast()
		{
			uint size = fd_.width * cd_.nsamples;
			fn_vect_.push_back([=]() {
				manual_contrast_correction(
					buffers_.gpu_float_cut_xz_,
					size,
					65535,
					cd_.contrast_min_slice_xz,
					cd_.contrast_max_slice_xz);
			});
			fn_vect_.push_back([=]() {
				manual_contrast_correction(
					buffers_.gpu_float_cut_yz_,
					size,
					65535,
					cd_.contrast_min_slice_yz,
					cd_.contrast_max_slice_yz);
			});
		}

		void Contrast::insert_autocontrast(std::atomic<bool>& autocontrast_request)
		{
			// requested check are inside the lambda so that we don't need to refresh the pipe at each autocontrast
			auto lambda_autocontrast = [&]() {
				if (autocontrast_request)
				{
					if (cd_.current_window == XYview)
					{
						if (cd_.vision_3d_enabled)
							autocontrast_caller(
								reinterpret_cast<float *>(gpu_3d_vision_->get_buffer()) + gpu_3d_vision_->get_pixels() * cd_.pindex,
								fd_.frame_res() * cd_.nsamples,
								0,
								XYview);
						else
							autocontrast_caller(
								buffers_.gpu_float_buffer_,
								buffers_.gpu_float_buffer_size_ / sizeof(float),
								0,
								XYview);
					}
					if (cd_.stft_view_enabled)
					{
						if (cd_.current_window == XZview)
							autocontrast_caller(
								buffers_.gpu_float_cut_xz_,
								fd_.width * cd_.nsamples,
								fd_.width * cd_.cuts_contrast_p_offset,
								XZview);
						else if (cd_.current_window == YZview)
							autocontrast_caller(
								buffers_.gpu_float_cut_yz_,
								fd_.width * cd_.nsamples,
								fd_.width * cd_.cuts_contrast_p_offset,
								YZview);
					}
					autocontrast_request = false;
				}
			};
			fn_vect_.push_back(lambda_autocontrast);
		}

		void Contrast::autocontrast_caller(float*			input,
			const uint			size,
			const uint			offset,
			WindowKind			view,
			cudaStream_t		stream)
		{
			float contrast_min = 0.f;
			float contrast_max = 0.f;
			auto_contrast_correction(input, size, offset, &contrast_min, &contrast_max, stream);
			switch (view)
			{
			case XYview:
				cd_.contrast_min_slice_xy = contrast_min;
				cd_.contrast_max_slice_xy = contrast_max;
				break;
			case YZview:
				cd_.contrast_min_slice_yz = contrast_min;
				cd_.contrast_max_slice_yz = contrast_max;
				break;
			case XZview:
				cd_.contrast_min_slice_xz = contrast_min;
				cd_.contrast_max_slice_xz = contrast_max;
				break;
			}
			cd_.notify_observers();
		}


		void Contrast::average_record_caller(
			float* input,
			const unsigned int width,
			const unsigned int height,
			const units::RectFd& signal,
			const units::RectFd& noise,
			cudaStream_t stream)
		{
			if (average_env_.average_n_ > 0)
			{
				average_env_.average_output_->push_back(make_average_plot(input, width, height, signal, noise, stream));
				average_env_.average_n_--;
			}
			else
			{
				average_env_.average_n_ = 0;
				average_env_.average_output_ = nullptr;
				Ic_->request_refresh();
			}
		}
	}
}