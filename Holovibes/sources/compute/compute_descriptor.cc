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
	using	LockGuard = std::lock_guard<std::mutex>;

	ComputeDescriptor::ComputeDescriptor() : Observable(),
		algorithm(Algorithm::None)
		, compute_mode(Computation::Direct)
		, nsamples(2)
		, pindex(1)
		, lambda(532e-9f)
		, zdistance(1.50f)
		, view_mode(ComplexViewMode::Modulus)
		, unwrap_history_size(1)
		, special_buffer_size(10)
		, log_scale_enabled(false)
		, log_scale_enabled_cut_xz(false)
		, log_scale_enabled_cut_yz(false)
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
		, img_acc_cutsXZ_enabled(false)
		, img_acc_cutsYZ_enabled(false)
		, img_acc_buffer_size(20)
		, img_acc_level(1)
		, img_acc_cutsXZ_level(1)
		, img_acc_cutsYZ_level(1)
		, p_accu_enabled(false)
		, p_accu_min_level(1)
		, p_accu_max_level(1)
		, stft_level(16)
		, stft_steps(1)
		, ref_diff_level(15)
		, ref_diff_enabled(false)
		, ref_sliding_enabled(false)
		, stft_view_enabled(false)
		, signal_trig_enabled(false)
		, stft_slice_cursor(QPoint(0, 0))
		, signal_zone(gui::Rectangle(10, 10))
		, noise_zone(gui::Rectangle(10, 10))
		, autofocus_zone(gui::Rectangle(10, 10))
		, stft_roi_zone(gui::Rectangle(10, 10))
		, current_window(WindowKind::MainDisplay)
	{
	}
	
	ComputeDescriptor& ComputeDescriptor::operator=(const ComputeDescriptor& cd)
	{
		algorithm.exchange(cd.algorithm.load());
		compute_mode.exchange(cd.compute_mode.load());
		nsamples.exchange(cd.nsamples.load());
		pindex.exchange(cd.pindex.load());
		lambda.exchange(cd.lambda.load());
		zdistance.exchange(cd.zdistance.load());
		view_mode.exchange(cd.view_mode.load());
		unwrap_history_size.exchange(cd.unwrap_history_size.load());
		special_buffer_size.exchange(cd.special_buffer_size.load());
		log_scale_enabled.exchange(cd.log_scale_enabled.load());
		log_scale_enabled_cut_xz.exchange(cd.log_scale_enabled_cut_xz.load());
		log_scale_enabled_cut_yz.exchange(cd.log_scale_enabled_cut_yz.load());
		shift_corners_enabled.exchange(cd.shift_corners_enabled.load());
		contrast_enabled.exchange(cd.contrast_enabled.load());
		vibrometry_enabled.exchange(cd.vibrometry_enabled.load());
		convolution_enabled.exchange(cd.convolution_enabled.load());
		flowgraphy_enabled.exchange(cd.flowgraphy_enabled.load());
		stft_enabled.exchange(cd.stft_enabled.load());
		filter_2d_enabled.exchange(cd.filter_2d_enabled.load());
		average_enabled.exchange(cd.average_enabled.load());
		contrast_min.exchange(cd.contrast_min.load());
		contrast_max.exchange(cd.contrast_max.load());
		contrast_min_slice_xz.exchange(cd.contrast_min_slice_xz.load());
		contrast_min_slice_yz.exchange(cd.contrast_min_slice_yz.load());
		contrast_max_slice_xz.exchange(cd.contrast_max_slice_xz.load());
		contrast_max_slice_yz.exchange(cd.contrast_max_slice_yz.load());
		vibrometry_q.exchange(cd.vibrometry_q.load());
		autofocus_size.exchange(cd.autofocus_size.load());
		convo_matrix_width.exchange(cd.convo_matrix_width.load());
		convo_matrix_height.exchange(cd.convo_matrix_height.load());
		convo_matrix_z.exchange(cd.convo_matrix_z.load());
		autofocus_z_min.exchange(cd.autofocus_z_min.load());
		autofocus_z_max.exchange(cd.autofocus_z_max.load());
		autofocus_z_div.exchange(cd.autofocus_z_div.load());
		autofocus_z_iter.exchange(cd.autofocus_z_iter.load());
		flowgraphy_level.exchange(cd.flowgraphy_level.load());
		is_cine_file.exchange(cd.is_cine_file.load());
		import_pixel_size.exchange(cd.import_pixel_size.load());
		img_acc_enabled.exchange(cd.img_acc_enabled.load());
		img_acc_cutsXZ_enabled.exchange(cd.img_acc_cutsXZ_enabled.load());
		img_acc_cutsYZ_enabled.exchange(cd.img_acc_cutsYZ_enabled.load());
		img_acc_buffer_size.exchange(cd.img_acc_buffer_size.load());
		img_acc_level.exchange(cd.img_acc_level.load());
		img_acc_cutsXZ_level.exchange(cd.img_acc_cutsXZ_level.load());
		img_acc_cutsYZ_level.exchange(cd.img_acc_cutsYZ_level.load());
		stft_level.exchange(cd.stft_level.load());
		stft_steps.exchange(cd.stft_steps.load());
		ref_diff_level.exchange(cd.ref_diff_level.load());
		ref_diff_enabled.exchange(cd.ref_diff_enabled.load());
		ref_sliding_enabled.exchange(cd.ref_sliding_enabled.load());
		stft_view_enabled.exchange(cd.stft_view_enabled.load());
		signal_trig_enabled.exchange(cd.signal_trig_enabled.load());
		current_window.exchange(cd.current_window.load());
		stft_slice_cursor = stft_slice_cursor;
		signal_zone = signal_zone;
		noise_zone = noise_zone;
		autofocus_zone = autofocus_zone;
		stft_roi_zone;
		return *this;
	}

	void ComputeDescriptor::stftCursor(QPoint *p, AccessMode m)
	{
		LockGuard g(mutex_);
		if (m == Get)
		{
			*p = stft_slice_cursor;
		}
		else if (m == Set)
		{
			stft_slice_cursor = *p;
		}
	}

	void ComputeDescriptor::signalZone(gui::Rectangle& rect, AccessMode m)
	{
		LockGuard g(mutex_);
		if (m == Get)
		{
			rect = signal_zone;
		}
		else if (m == Set)
		{
			signal_zone = rect;
		}
	}

	void ComputeDescriptor::noiseZone(gui::Rectangle& rect, AccessMode m)
	{
		LockGuard g(mutex_);
		if (m == Get)
		{
			rect = noise_zone;
		}
		else if (m == Set)
		{
			noise_zone = rect;
		}
	}

	void ComputeDescriptor::autofocusZone(gui::Rectangle& rect, AccessMode m)
	{
		LockGuard g(mutex_);
		if (m == Get)
		{
			rect = autofocus_zone;
		}
		else if (m == Set)
		{
			autofocus_zone = rect;
		}
	}

	void ComputeDescriptor::stftRoiZone(gui::Rectangle& rect, AccessMode m)
	{
		LockGuard g(mutex_);
		if (m == Get)
		{
			rect = stft_roi_zone;
		}
		else if (m == Set)
		{
			stft_roi_zone = rect;
		}
	}


}