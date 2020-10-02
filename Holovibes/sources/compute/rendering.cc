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

#include "rendering.hh"
#include "frame_desc.hh"
#include "icompute.hh"
#include "compute_descriptor.hh"
#include "concurrent_deque.hh"
#include "contrast_correction.cuh"
#include "average.cuh"
#include "stft.cuh"

namespace holovibes
{
	namespace compute
	{
		Rendering::Rendering(FnVector& fn_vect,
			const CoreBuffers& buffers,
			Average_env& average_env,
			ComputeDescriptor& cd,
			const camera::FrameDescriptor& input_fd,
			const camera::FrameDescriptor& output_fd,
			ICompute* Ic)
			: fn_vect_(fn_vect)
			, buffers_(buffers)
			, average_env_(average_env)
			, cd_(cd)
			, input_fd_(input_fd)
			, fd_(output_fd)
			, Ic_(Ic)
		{
		}

		void Rendering::insert_fft_shift()
		{
			if (cd_.fft_shift_enabled)
			{
				if (cd_.img_type == ImgType::Composite)
					fn_vect_.push_back([=]() {
						shift_corners(
							reinterpret_cast<float3 *>(buffers_.gpu_float_buffer_.get()),
							fd_.width,
							fd_.height);
					});
				else
					fn_vect_.push_back([=]() {
						shift_corners(
							buffers_.gpu_float_buffer_,
							fd_.width,
							fd_.height);
					});
			}
		}

		void Rendering::insert_average(std::atomic<bool>& record_request)
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

		void Rendering::insert_log()
		{
			if (cd_.log_scale_slice_xy_enabled)
				insert_main_log();
			if (cd_.stft_view_enabled)
				insert_slice_log();
		}

		void Rendering::insert_contrast(std::atomic<bool>& autocontrast_request, std::atomic<bool>& autocontrast_slice_xz_request, std::atomic<bool>& autocontrast_slice_yz_request)
		{
			insert_autocontrast(autocontrast_request, autocontrast_slice_xz_request, autocontrast_slice_yz_request);
			if (cd_.contrast_enabled)
				insert_main_contrast();

			if (cd_.stft_view_enabled)
				insert_slice_contrast();
		}

		//----------

		void Rendering::insert_main_average()
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

		void Rendering::insert_average_record()
		{
			units::RectFd signalZone;
			units::RectFd noiseZone;
			cd_.signalZone(signalZone, AccessMode::Get);
			cd_.noiseZone(noiseZone, AccessMode::Get);
			fn_vect_.push_back([=]() {average_record_caller(signalZone, noiseZone); });
		}

		void Rendering::insert_main_log()
		{
			fn_vect_.push_back([=]() {apply_log10(buffers_.gpu_float_buffer_, buffers_.gpu_float_buffer_size_); });
		}

		void Rendering::insert_slice_log()
		{
			uint size = fd_.width * cd_.nSize;
			if (cd_.log_scale_slice_xz_enabled)
				fn_vect_.push_back([=]() {apply_log10(static_cast<float *>(buffers_.gpu_float_cut_xz_.get()), size); });
			if (cd_.log_scale_slice_yz_enabled)
				fn_vect_.push_back([=]() {apply_log10(static_cast<float *>(buffers_.gpu_float_cut_yz_.get()), size); });
		}

		void Rendering::insert_main_contrast()
		{
			uint size = buffers_.gpu_float_buffer_size_;
			fn_vect_.push_back([=]() {
				manual_contrast_correction(
					buffers_.gpu_float_buffer_,
					size,
					65535,
					cd_.contrast_invert ? cd_.contrast_max_slice_xy : cd_.contrast_min_slice_xy,
					cd_.contrast_invert ? cd_.contrast_min_slice_xy : cd_.contrast_max_slice_xy);
			});
		}

		void Rendering::insert_slice_contrast()
		{
			uint size = fd_.width * cd_.nSize;
			fn_vect_.push_back([=]() {
				manual_contrast_correction(
					static_cast<float *>(buffers_.gpu_float_cut_xz_.get()),
					size,
					65535,
					cd_.contrast_invert ? cd_.contrast_max_slice_xz : cd_.contrast_min_slice_xz,
					cd_.contrast_invert ? cd_.contrast_min_slice_xz : cd_.contrast_max_slice_xz);
			});
			fn_vect_.push_back([=]() {
				manual_contrast_correction(
					static_cast<float *>(buffers_.gpu_float_cut_yz_.get()),
					size,
					65535,
					cd_.contrast_invert ? cd_.contrast_max_slice_yz : cd_.contrast_min_slice_yz,
					cd_.contrast_invert ? cd_.contrast_min_slice_yz : cd_.contrast_max_slice_yz);
			});
		}

		void Rendering::insert_autocontrast(std::atomic<bool>& autocontrast_request, std::atomic<bool>& autocontrast_slice_xz_request, std::atomic<bool>& autocontrast_slice_yz_request)
		{
			// requested check are inside the lambda so that we don't need to refresh the pipe at each autocontrast
			auto lambda_autocontrast = [&]() {
				if (autocontrast_request)
					autocontrast_caller(
						buffers_.gpu_float_buffer_,
						buffers_.gpu_float_buffer_size_,
						0,
						XYview);
				if (autocontrast_slice_xz_request)
					autocontrast_caller(
						static_cast<float *>(buffers_.gpu_float_cut_xz_.get()),
						fd_.width * cd_.nSize,
						fd_.width * cd_.cuts_contrast_p_offset,
						XZview);
				if (autocontrast_slice_yz_request)
					autocontrast_caller(
						static_cast<float *>(buffers_.gpu_float_cut_yz_.get()),
						fd_.width * cd_.nSize,
						fd_.width * cd_.cuts_contrast_p_offset,
						YZview);
				autocontrast_request = false;
				autocontrast_slice_xz_request = false;
				autocontrast_slice_yz_request = false;
			};
			fn_vect_.push_back(lambda_autocontrast);
		}

		void Rendering::autocontrast_caller(float*			input,
			const uint			size,
			const uint			offset,
			WindowKind			view,
			cudaStream_t		stream)
		{
			float contrast_min = 0.f;
			float contrast_max = 0.f;
			auto_contrast_correction(input, size, offset, &contrast_min, &contrast_max,
				cd_.contrast_threshold_low_percentile, cd_.contrast_threshold_high_percentile);
			switch (view)
			{
			case XYview:
				cd_.contrast_min_slice_xy = contrast_min;
				cd_.contrast_max_slice_xy = contrast_max;
				break;
			case YZview:
				/*
				In order to make YZview work follow this:
					- rotate 90 degrees  YZview image
					- compute min max with offset
					- rotate -90 degrees

				cd_.contrast_min_slice_xz = contrast_min;
				cd_.contrast_max_slice_xz = contrast_max;

				*/
				break;
			case XZview:
				// temporary hack
				cd_.contrast_min_slice_xz = contrast_min;
				cd_.contrast_max_slice_xz = contrast_max;
				cd_.contrast_min_slice_yz = contrast_min;
				cd_.contrast_max_slice_yz = contrast_max;
				break;
			}
			cd_.notify_observers();
		}


		void Rendering::average_record_caller(
			const units::RectFd& signal,
			const units::RectFd& noise,
			cudaStream_t stream)
		{
			if (average_env_.average_n_ > 0)
			{
				average_env_.average_output_->push_back(make_average_plot(buffers_.gpu_float_buffer_, input_fd_.width, input_fd_.height, signal, noise, stream));
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
