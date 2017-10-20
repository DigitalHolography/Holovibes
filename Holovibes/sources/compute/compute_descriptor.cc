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
		stabilization_zone(2048, 2048),	//temporary
		algorithm(Algorithm::None),
		compute_mode(Computation::Stop),
		nsamples(1),
		pindex(0),
		lambda(532e-9f),
		zdistance(1.50f),
		img_type(ImgType::Modulus),
		unwrap_history_size(1),
		special_buffer_size(10),
		log_scale_slice_xy_enabled(false),
		log_scale_slice_xz_enabled(false),
		log_scale_slice_yz_enabled(false),
		shift_corners_enabled(false),
		contrast_enabled(false),
		vibrometry_enabled(false),
		convolution_enabled(false),
		flowgraphy_enabled(false),
		stft_enabled(false),
		filter_2d_enabled(false),
		average_enabled(false),
		contrast_min_slice_xy(1.f),
		contrast_max_slice_xy(65535.f),
		contrast_min_slice_xz(1.f),
		contrast_min_slice_yz(1.f),
		contrast_max_slice_xz(65535.f),
		contrast_max_slice_yz(65535.f),
		vibrometry_q(0),
		autofocus_size(3),
		convo_matrix_width(0),
		convo_matrix_height(0),
		convo_matrix_z(0),
		autofocus_z_min(0.f),
		autofocus_z_max(1.f),
		autofocus_z_div(10),
		autofocus_z_iter(3),
		flowgraphy_level(3),
		is_cine_file(false),
		pixel_size(5.42f),
		img_acc_slice_xy_enabled(false),
		img_acc_slice_xz_enabled(false),
		img_acc_slice_yz_enabled(false),
		img_acc_slice_xy_level(1),
		img_acc_slice_xz_level(1),
		img_acc_slice_yz_level(1),
		p_accu_enabled(false),
		p_accu_min_level(1),
		p_accu_max_level(1),
		x_accu_enabled(false),
		x_accu_min_level(1),
		x_accu_max_level(1),
		y_accu_enabled(false),
		y_accu_min_level(1),
		y_accu_max_level(1),
		stft_level(16),
		stft_steps(1),
		ref_diff_level(15),
		ref_diff_enabled(false),
		ref_sliding_enabled(false),
		stft_view_enabled(false),
		stft_slice_cursor(QPoint(0, 0)),
		signal_zone(gui::Rectangle(0, 0)),
		noise_zone(gui::Rectangle(0, 0)),
		autofocus_zone(gui::Rectangle(0, 0)),
		stft_roi_zone(gui::Rectangle(0, 0)),
		current_window(WindowKind::XYview),
		cuts_contrast_p_offset(4),
		vision_3d_enabled(false),
		display_rate(30)
	{

	}

	ComputeDescriptor::~ComputeDescriptor()
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
		img_type.exchange(cd.img_type.load());
		unwrap_history_size.exchange(cd.unwrap_history_size.load());
		special_buffer_size.exchange(cd.special_buffer_size.load());
		log_scale_slice_xy_enabled.exchange(cd.log_scale_slice_xy_enabled.load());
		log_scale_slice_xz_enabled.exchange(cd.log_scale_slice_xz_enabled.load());
		log_scale_slice_yz_enabled.exchange(cd.log_scale_slice_yz_enabled.load());
		shift_corners_enabled.exchange(cd.shift_corners_enabled.load());
		contrast_enabled.exchange(cd.contrast_enabled.load());
		vibrometry_enabled.exchange(cd.vibrometry_enabled.load());
		convolution_enabled.exchange(cd.convolution_enabled.load());
		flowgraphy_enabled.exchange(cd.flowgraphy_enabled.load());
		stft_enabled.exchange(cd.stft_enabled.load());
		filter_2d_enabled.exchange(cd.filter_2d_enabled.load());
		average_enabled.exchange(cd.average_enabled.load());
		contrast_min_slice_xy.exchange(cd.contrast_min_slice_xy.load());
		contrast_max_slice_xy.exchange(cd.contrast_max_slice_xy.load());
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
		pixel_size.exchange(cd.pixel_size.load());
		img_acc_slice_xy_enabled.exchange(cd.img_acc_slice_xy_enabled.load());
		img_acc_slice_xz_enabled.exchange(cd.img_acc_slice_xz_enabled.load());
		img_acc_slice_yz_enabled.exchange(cd.img_acc_slice_yz_enabled.load());
		img_acc_slice_xy_level.exchange(cd.img_acc_slice_xy_level.load());
		img_acc_slice_xz_level.exchange(cd.img_acc_slice_xz_level.load());
		img_acc_slice_yz_level.exchange(cd.img_acc_slice_yz_level.load());
		stft_level.exchange(cd.stft_level.load());
		stft_steps.exchange(cd.stft_steps.load());
		ref_diff_level.exchange(cd.ref_diff_level.load());
		ref_diff_enabled.exchange(cd.ref_diff_enabled.load());
		ref_sliding_enabled.exchange(cd.ref_sliding_enabled.load());
		stft_view_enabled.exchange(cd.stft_view_enabled.load());
		current_window.exchange(cd.current_window.load());
		cuts_contrast_p_offset.exchange(cd.cuts_contrast_p_offset.load());
		vision_3d_enabled.exchange(cd.vision_3d_enabled.load());
		display_rate.exchange(cd.display_rate.load());
		stft_slice_cursor = cd.stft_slice_cursor;
		signal_zone = cd.signal_zone;
		noise_zone = cd.noise_zone;
		autofocus_zone = cd.autofocus_zone;
		stft_roi_zone = cd.stft_roi_zone;
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
	gui::Rectangle ComputeDescriptor::getStabilizationZone() const
	{
		LockGuard g(mutex_);
		return stabilization_zone;
	}

	void ComputeDescriptor::setStabilizationZone(const gui::Rectangle& rect)
	{
		LockGuard g(mutex_);
		stabilization_zone = rect;
	}
}