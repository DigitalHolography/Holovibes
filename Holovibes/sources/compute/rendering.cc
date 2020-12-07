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
#include "chart.cuh"
#include "stft.cuh"
#include "percentile.cuh"

namespace holovibes
{
	namespace compute
	{
		Rendering::Rendering(FunctionVector& fn_compute_vect,
			const CoreBuffersEnv& buffers,
			ChartEnv& chart_env,
			const ImageAccEnv& image_acc_env,
			const TimeTransformationEnv& time_transformation_env,
			ComputeDescriptor& cd,
			const camera::FrameDescriptor& input_fd,
			const camera::FrameDescriptor& output_fd,
			ICompute* Ic)
			: fn_compute_vect_(fn_compute_vect)
			, buffers_(buffers)
			, chart_env_(chart_env)
			, image_acc_env_(image_acc_env)
			, time_transformation_env_(time_transformation_env)
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
					fn_compute_vect_.conditional_push_back([=]() {
						shift_corners(
							reinterpret_cast<float3 *>(buffers_.gpu_postprocess_frame.get()),
							1,
							fd_.width,
							fd_.height);
					});
				else
					fn_compute_vect_.conditional_push_back([=]() {
						shift_corners(
							buffers_.gpu_postprocess_frame,
							1,
							fd_.width,
							fd_.height);
					});
			}
		}

		void Rendering::insert_chart()
		{
			if (cd_.chart_display_enabled || cd_.chart_record_enabled)
			{
				fn_compute_vect_.conditional_push_back([=]() {
					units::RectFd signal_zone;
					units::RectFd noise_zone;
					cd_.signalZone(signal_zone, AccessMode::Get);
					cd_.noiseZone(noise_zone, AccessMode::Get);

					if (signal_zone.width() == 0 || signal_zone.height() == 0
						|| noise_zone.width() == 0 || noise_zone.height() == 0)
						return;

					ChartPoint point = make_chart_plot(
							buffers_.gpu_postprocess_frame,
							input_fd_.width,
							input_fd_.height,
							signal_zone,
							noise_zone);

					if (cd_.chart_display_enabled)
						chart_env_.chart_display_queue_->push_back(point);
					if (cd_.chart_record_enabled && chart_env_.nb_chart_points_to_record_ != 0)
					{
						chart_env_.chart_record_queue_->push_back(point);
						--chart_env_.nb_chart_points_to_record_;
					}
				});
			}
		}

		void Rendering::insert_log()
		{
			if (cd_.log_scale_slice_xy_enabled)
				insert_main_log();
			if (cd_.time_transformation_cuts_enabled)
				insert_slice_log();
		}

		void Rendering::insert_contrast(std::atomic<bool>& autocontrast_request,
									    std::atomic<bool>& autocontrast_slice_xz_request,
										std::atomic<bool>& autocontrast_slice_yz_request)
		{
			// Do not compute contrast or apply contrast if not enabled
			if (!cd_.contrast_enabled)
				return;

			// Compute min and max pixel values if requested
			insert_compute_autocontrast(autocontrast_request,
				autocontrast_slice_xz_request, autocontrast_slice_yz_request);

			// Apply contrast on the main view
			insert_apply_contrast(WindowKind::XYview);

			// Apply contrast on cuts if needed
			if (cd_.time_transformation_cuts_enabled)
			{
				insert_apply_contrast(WindowKind::XZview);
				insert_apply_contrast(WindowKind::YZview);
			}
		}

		void Rendering::insert_main_log()
		{
			fn_compute_vect_.conditional_push_back([=]() {apply_log10(buffers_.gpu_postprocess_frame, buffers_.gpu_postprocess_frame_size); });
		}

		void Rendering::insert_slice_log()
		{
			if (cd_.log_scale_slice_xz_enabled)
				fn_compute_vect_.conditional_push_back([=]() {apply_log10(buffers_.gpu_postprocess_frame_xz.get(), fd_.width * cd_.time_transformation_size); });
			if (cd_.log_scale_slice_yz_enabled)
				fn_compute_vect_.conditional_push_back([=]() {apply_log10(buffers_.gpu_postprocess_frame_yz.get(), fd_.height * cd_.time_transformation_size); });
		}

		void Rendering::insert_apply_contrast(WindowKind view)
		{
			fn_compute_vect_.conditional_push_back([=](){
				// Set parameters
				float* input = nullptr;
				uint size = 0;
				constexpr ushort dynamic_range = 65535;
				float min = 0;
				float max = 0;

				switch (view)
				{
				case XYview:
					input = buffers_.gpu_postprocess_frame;
					size = buffers_.gpu_postprocess_frame_size;
					min = cd_.contrast_invert ? cd_.contrast_max_slice_xy : cd_.contrast_min_slice_xy;
					max = cd_.contrast_invert ? cd_.contrast_min_slice_xy : cd_.contrast_max_slice_xy;
					break;
				case YZview:
					input = buffers_.gpu_postprocess_frame_yz.get();
					size = fd_.height * cd_.time_transformation_size;
					min = cd_.contrast_invert ? cd_.contrast_max_slice_yz : cd_.contrast_min_slice_yz;
					max = cd_.contrast_invert ? cd_.contrast_min_slice_yz : cd_.contrast_max_slice_yz;
					break;
				case XZview:
					input = buffers_.gpu_postprocess_frame_xz.get();
					size = fd_.width * cd_.time_transformation_size;
					min = cd_.contrast_invert ? cd_.contrast_max_slice_xz : cd_.contrast_min_slice_xz;
					max = cd_.contrast_invert ? cd_.contrast_min_slice_xz : cd_.contrast_max_slice_xz;
					break;
				}

				apply_contrast_correction(input, size, dynamic_range, min, max);
			});
		}

		void Rendering::insert_compute_autocontrast(std::atomic<bool>& autocontrast_request,
													std::atomic<bool>& autocontrast_slice_xz_request,
													std::atomic<bool>& autocontrast_slice_yz_request)
		{
			// requested check are inside the lambda so that we don't need to
			// refresh the pipe at each autocontrast
			auto lambda_autocontrast = [&]() {
				// Compute autocontrast once the gpu time transformation queue is full
				if (!time_transformation_env_.gpu_time_transformation_queue->is_full())
					return;

				if (autocontrast_request && (!image_acc_env_.gpu_accumulation_xy_queue ||
					image_acc_env_.gpu_accumulation_xy_queue->is_full()))
				{
					// FIXME Handle composite size, adapt width and height (frames_res = buffers_.gpu_postprocess_frame_size)
					autocontrast_caller(
						buffers_.gpu_postprocess_frame,
						fd_.width,
						fd_.height,
						0,
						XYview);
					autocontrast_request = false;
				}
				if (autocontrast_slice_xz_request && (!image_acc_env_.gpu_accumulation_xz_queue ||
					image_acc_env_.gpu_accumulation_xz_queue->is_full()))
				{
					autocontrast_caller(
						buffers_.gpu_postprocess_frame_xz.get(),
						fd_.width,
						cd_.time_transformation_size,
						cd_.cuts_contrast_p_offset,
						XZview);
					autocontrast_slice_xz_request = false;
				}
				if (autocontrast_slice_yz_request && (!image_acc_env_.gpu_accumulation_yz_queue ||
					image_acc_env_.gpu_accumulation_yz_queue->is_full()))
				{
					autocontrast_caller(
						buffers_.gpu_postprocess_frame_yz.get(),
						cd_.time_transformation_size,
						fd_.height,
						cd_.cuts_contrast_p_offset,
						YZview);
					autocontrast_slice_yz_request = false;
				}
			};

			fn_compute_vect_.conditional_push_back(lambda_autocontrast);
		}

		void Rendering::set_contrast_min_max(const float* const percent_out,
											 std::atomic<float>& contrast_min,
											 std::atomic<float>& contrast_max)
		{
			contrast_min = percent_out[0];
			contrast_max = percent_out[1];

			contrast_min = ((contrast_min < 1.0f) ? (1.0f) : contrast_min);
			contrast_max = ((contrast_max < 1.0f) ? (1.0f) : contrast_max);
		}

		void Rendering::autocontrast_caller(float*				input,
											const uint			width,
											const uint			height,
											const uint			offset,
											WindowKind			view,
											cudaStream_t		stream)
		{
			constexpr uint percent_size = 2;

			const float percent_in[percent_size] = { cd_.contrast_lower_threshold,
											cd_.contrast_upper_threshold };
			float percent_out[percent_size] = { -1 };

			switch (view)
			{
			case XYview:
				// No offset
				compute_percentile_xy_view(input, width, height, percent_in, percent_out,
					percent_size, cd_.getReticleZone(), cd_.reticle_enabled);
				set_contrast_min_max(percent_out, cd_.contrast_min_slice_xy, cd_.contrast_max_slice_xy);
				break;
			case YZview:
				compute_percentile_yz_view(input, width, height, offset, percent_in, percent_out,
					percent_size, cd_.getReticleZone(), cd_.reticle_enabled);
				set_contrast_min_max(percent_out, cd_.contrast_min_slice_yz, cd_.contrast_max_slice_yz);
				break;
			case XZview:
				compute_percentile_xz_view(input, width, height, offset, percent_in, percent_out,
					percent_size, cd_.getReticleZone(), cd_.reticle_enabled);
				set_contrast_min_max(percent_out, cd_.contrast_min_slice_xz, cd_.contrast_max_slice_xz);
				break;
			}

			cd_.notify_observers();
		}
	}
}
