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

#include "compute_descriptor.hh"

namespace holovibes
{
	ComputeDescriptor::ComputeDescriptor() : Observable(),
		algorithm(fft_algorithm::None)
		, compute_mode(compute_mode::DIRECT)
		, nsamples(2)
		, pindex(1)
		, lambda(532e-9f)
		, zdistance(1.50f)
		, view_mode(complex_view_mode::MODULUS)
		, unwrap_history_size(1)
		, special_buffer_size(10)
		, log_scale_enabled(false)
		, shift_corners_enabled(true)
		, contrast_enabled(false)
		, vibrometry_enabled(false)
		, convolution_enabled(false)
		, flowgraphy_enabled(false)
		, stft_enabled(false)
		, filter_2d_enabled(false)
		, average_enabled(false)
		, contrast_min(1.f)
		, contrast_max(65535.f)
		, contrast_min_slice_xz(1.f)
		, contrast_min_slice_yz(1.f)
		, contrast_max_slice_xz(65535.f)
		, contrast_max_slice_yz(65535.f)
		, vibrometry_q(0)
		, autofocus_size(3)
		, convo_matrix_width(0)
		, convo_matrix_height(0)
		, convo_matrix_z(0)
		, autofocus_z_min(0.f)
		, autofocus_z_max(1.f)
		, autofocus_z_div(10)
		, autofocus_z_iter(3)
		, flowgraphy_level(3)
		, is_cine_file(false)
		, import_pixel_size(5.42f)
		, img_acc_enabled(false)
		, img_acc_buffer_size(20)
		, img_acc_level(1)
		, stft_level(16)
		, stft_steps(1)
		, ref_diff_level(15)
		, ref_diff_enabled(false)
		, ref_sliding_enabled(false)
		, stft_view_enabled(false)
		, signal_trig_enabled(false)
		, stft_slice_cursor(QPoint(0, 0))
		, signal_zone(Rectangle(10, 10))
		, noise_zone(Rectangle(10, 10))
		, autofocus_zone(Rectangle(10, 10))
		, stft_roi_zone(Rectangle(10, 10))
		, current_window(window::MAIN_DISPLAY)
	{
	}

	ComputeDescriptor& ComputeDescriptor::operator=(const ComputeDescriptor& cd)
	{
		compute_mode.exchange(cd.compute_mode.load());
		algorithm.exchange(cd.algorithm.load());
		nsamples.exchange(cd.nsamples.load());
		pindex.exchange(cd.pindex.load());
		lambda.exchange(cd.lambda.load());
		zdistance.exchange(cd.zdistance.load());
		view_mode.exchange(cd.view_mode.load());
		unwrap_history_size.exchange(cd.unwrap_history_size.load());
		log_scale_enabled.exchange(cd.log_scale_enabled.load());
		shift_corners_enabled.exchange(cd.shift_corners_enabled.load());
		contrast_enabled.exchange(cd.contrast_enabled.load());
		vibrometry_enabled.exchange(cd.vibrometry_enabled.load());
		contrast_min.exchange(cd.contrast_min.load());
		contrast_max.exchange(cd.contrast_max.load());
		contrast_min_slice_xz.exchange(cd.contrast_min_slice_xz.load());
		contrast_min_slice_yz.exchange(cd.contrast_min_slice_yz.load());
		contrast_max_slice_xz.exchange(cd.contrast_max_slice_xz.load());
		contrast_max_slice_yz.exchange(cd.contrast_max_slice_yz.load());
		vibrometry_q.exchange(cd.vibrometry_q.load());
		autofocus_size.exchange(cd.autofocus_size.load());
		stft_enabled.exchange(cd.stft_enabled.load());
		//TODO : to complete
		return *this;
	}

	void ComputeDescriptor::stftCursor(QPoint *p, t_access mode)
	{
		guard g(mutex_);
		if (mode == Get)
		{
			*p = stft_slice_cursor;
		}
		else if (mode == Set)
		{
			stft_slice_cursor = *p;
		}
	}

	void ComputeDescriptor::signalZone(Rectangle *rect, t_access mode)
	{
		guard g(mutex_);
		if (mode == Get)
		{
			*rect = signal_zone;
		}
		else if (mode == Set)
		{
			signal_zone = *rect;
		}
	}

	void ComputeDescriptor::noiseZone(Rectangle *rect, t_access mode)
	{
		guard g(mutex_);
		if (mode == Get)
		{
			*rect = noise_zone;
		}
		else if (mode == Set)
		{
			noise_zone = *rect;
		}
	}

	void ComputeDescriptor::autofocusZone(Rectangle *rect, t_access mode)
	{
		guard g(mutex_);
		if (mode == Get)
		{
			*rect = autofocus_zone;
		}
		else if (mode == Set)
		{
			autofocus_zone = *rect;
		}
	}

	void ComputeDescriptor::stftRoiZone(Rectangle *rect, t_access mode)
	{
		guard g(mutex_);
		if (mode == Get)
		{
			*rect = stft_roi_zone;
		}
		else if (mode == Set)
		{
			stft_roi_zone = *rect;
		}
	}


}