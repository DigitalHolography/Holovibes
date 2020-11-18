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

	ComputeDescriptor::ComputeDescriptor() : Observable()
	{}

	ComputeDescriptor::~ComputeDescriptor()
	{}

	ComputeDescriptor& ComputeDescriptor::operator=(const ComputeDescriptor& cd)
	{
		space_transformation = cd.space_transformation.load();
		time_transformation = cd.time_transformation.load();
		compute_mode = cd.compute_mode.load();
		time_transformation_size = cd.time_transformation_size.load();
		pindex = cd.pindex.load();
		lambda = cd.lambda.load();
		zdistance = cd.zdistance.load();
		img_type = cd.img_type.load();
		unwrap_history_size = cd.unwrap_history_size.load();
		log_scale_slice_xy_enabled = cd.log_scale_slice_xy_enabled.load();
		log_scale_slice_xz_enabled = cd.log_scale_slice_xz_enabled.load();
		log_scale_slice_yz_enabled = cd.log_scale_slice_yz_enabled.load();
		fft_shift_enabled = cd.fft_shift_enabled.load();
		contrast_enabled = cd.contrast_enabled.load();
		convolution_enabled = cd.convolution_enabled.load();
		chart_display_enabled = cd.chart_display_enabled.load();
		chart_record_enabled = cd.chart_record_enabled.load();
		contrast_min_slice_xy = cd.contrast_min_slice_xy.load();
		contrast_max_slice_xy = cd.contrast_max_slice_xy.load();
		contrast_min_slice_xz = cd.contrast_min_slice_xz.load();
		contrast_min_slice_yz = cd.contrast_min_slice_yz.load();
		contrast_max_slice_xz = cd.contrast_max_slice_xz.load();
		contrast_max_slice_yz = cd.contrast_max_slice_yz.load();
		contrast_invert = cd.contrast_invert.load();
		convo_matrix_width = cd.convo_matrix_width.load();
		convo_matrix_height = cd.convo_matrix_height.load();
		convo_matrix_z = cd.convo_matrix_z.load();
		pixel_size = cd.pixel_size.load();
		img_acc_slice_xy_enabled = cd.img_acc_slice_xy_enabled.load();
		img_acc_slice_xz_enabled = cd.img_acc_slice_xz_enabled.load();
		img_acc_slice_yz_enabled = cd.img_acc_slice_yz_enabled.load();
		img_acc_slice_xy_level = cd.img_acc_slice_xy_level.load();
		img_acc_slice_xz_level = cd.img_acc_slice_xz_level.load();
		img_acc_slice_yz_level = cd.img_acc_slice_yz_level.load();
		time_transformation_stride = cd.time_transformation_stride.load();
		time_transformation_cuts_enabled = cd.time_transformation_cuts_enabled.load();
		current_window = cd.current_window.load();
		cuts_contrast_p_offset = cd.cuts_contrast_p_offset.load();
		display_rate = cd.display_rate.load();
		reticle_enabled = cd.reticle_enabled.load();
		reticle_scale = cd.reticle_scale.load();
		stft_slice_cursor = cd.stft_slice_cursor;
		signal_zone = cd.signal_zone;
		noise_zone = cd.noise_zone;
		stft_roi_zone = cd.stft_roi_zone;
		filter2D_sub_zone = cd.filter2D_sub_zone;
		contrast_auto_refresh = cd.contrast_auto_refresh.load();
		request_recorder_copy_frames = cd.request_recorder_copy_frames.load();
		nb_frames_record = cd.nb_frames_record.load();
		copy_frames_done = cd.copy_frames_done.load();
		is_recording = cd.is_recording.load();
		return *this;
	}

	units::PointFd ComputeDescriptor::getStftCursor() const
	{
		LockGuard g(mutex_);
		return stft_slice_cursor;
	}

	void ComputeDescriptor::setStftCursor(const units::PointFd& rect)
	{
		LockGuard g(mutex_);
		stft_slice_cursor = rect;
	}

	void ComputeDescriptor::signalZone(units::RectFd& rect, AccessMode m)
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

	void ComputeDescriptor::noiseZone(units::RectFd& rect, AccessMode m)
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

	units::RectFd ComputeDescriptor::getStftZone() const
	{
		LockGuard g(mutex_);
		return stft_roi_zone;
	}

	void ComputeDescriptor::setStftZone(const units::RectFd& rect)
	{
		LockGuard g(mutex_);
		stft_roi_zone = rect;
	}

	units::RectFd ComputeDescriptor::getFilter2DSubZone() const
	{
		LockGuard g(mutex_);
		return filter2D_sub_zone;
	}

	void ComputeDescriptor::setFilter2DSubZone(const units::RectFd& rect)
	{
		LockGuard g(mutex_);
		filter2D_sub_zone = rect;
	}

	units::RectFd ComputeDescriptor::getCompositeZone() const
	{
		LockGuard g(mutex_);
		return composite_zone;
	}

	void ComputeDescriptor::setCompositeZone(const units::RectFd& rect)
	{
		LockGuard g(mutex_);
		composite_zone = rect;
	}

	units::RectFd ComputeDescriptor::getZoomedZone() const
	{
		LockGuard g(mutex_);
		return zoomed_zone;
	}

	void ComputeDescriptor::setZoomedZone(const units::RectFd& rect)
	{
		LockGuard g(mutex_);
		zoomed_zone = rect;
	}

	float ComputeDescriptor::get_contrast_min(WindowKind kind) const
	{
		switch (kind)
		{
		case WindowKind::XYview:
			return log_scale_slice_xy_enabled ? contrast_min_slice_xy.load() : log10(contrast_min_slice_xy);
		case WindowKind::XZview:
			return log_scale_slice_xz_enabled ? contrast_min_slice_xz.load() : log10(contrast_min_slice_xz);
		case WindowKind::YZview:
			return log_scale_slice_yz_enabled ? contrast_min_slice_yz.load() : log10(contrast_min_slice_yz);
		}
		return 0;
	}

	float ComputeDescriptor::get_contrast_max(WindowKind kind) const
	{
		switch (kind)
		{
		case WindowKind::XYview:
			return log_scale_slice_xy_enabled ? contrast_max_slice_xy.load() : log10(contrast_max_slice_xy);
		case WindowKind::XZview:
			return log_scale_slice_xz_enabled ? contrast_max_slice_xz.load() : log10(contrast_max_slice_xz);
		case WindowKind::YZview:
			return log_scale_slice_yz_enabled ? contrast_max_slice_yz.load() : log10(contrast_max_slice_yz);
		}
		return 0;
	}

	float ComputeDescriptor::get_truncate_contrast_min(WindowKind kind,
													   const int precision) const
	{
		float value = get_contrast_min(kind);
		const double multiplier = std::pow(10.0, precision);
		return std::round(value * multiplier) / multiplier;
	}

	float ComputeDescriptor::get_truncate_contrast_max(WindowKind kind,
													   const int precision) const
	{
		float value = get_contrast_max(kind);
		const double multiplier = std::pow(10.0, precision);
		return std::round(value * multiplier) / multiplier;
	}

	bool ComputeDescriptor::get_img_log_scale_slice_enabled(WindowKind kind) const
	{
		switch (kind)
		{
		case WindowKind::XYview:
			return log_scale_slice_xy_enabled;
		case WindowKind::XZview:
			return log_scale_slice_xz_enabled;
		case WindowKind::YZview:
			return log_scale_slice_yz_enabled;
		}
		return false;
	}

	bool ComputeDescriptor::get_img_acc_slice_enabled(WindowKind kind) const
	{
		switch (kind)
		{
		case WindowKind::XYview:
			return img_acc_slice_xy_enabled;
		case WindowKind::XZview:
			return img_acc_slice_xz_enabled;
		case WindowKind::YZview:
			return img_acc_slice_yz_enabled;
		}
		return false;
	}

	unsigned ComputeDescriptor::get_img_acc_slice_level(WindowKind kind) const
	{
		switch (kind)
		{
		case WindowKind::XYview:
			return img_acc_slice_xy_level;
		case WindowKind::XZview:
			return img_acc_slice_xz_level;
		case WindowKind::YZview:
			return img_acc_slice_yz_level;
		}
		return 0;
	}

	void ComputeDescriptor::set_contrast_min(WindowKind kind, float value)
	{
		switch (kind)
		{
		case WindowKind::XYview:
			contrast_min_slice_xy = log_scale_slice_xy_enabled ? value : pow(10, value);
			break;
		case WindowKind::XZview:
			contrast_min_slice_xz = log_scale_slice_xz_enabled ? value : pow(10, value);
			break;
		case WindowKind::YZview:
			contrast_min_slice_yz = log_scale_slice_yz_enabled ? value : pow(10, value);
			break;
		}
	}

	void ComputeDescriptor::set_contrast_max(WindowKind kind, float value)
	{
		switch (kind)
		{
		case WindowKind::XYview:
			contrast_max_slice_xy = log_scale_slice_xy_enabled ? value : pow(10, value);
			break;
		case WindowKind::XZview:
			contrast_max_slice_xz = log_scale_slice_xz_enabled ? value : pow(10, value);
			break;
		case WindowKind::YZview:
			contrast_max_slice_yz = log_scale_slice_yz_enabled ? value : pow(10, value);
			break;
		}
	}

	void ComputeDescriptor::set_log_scale_slice_enabled(WindowKind kind, bool value)
	{
		switch (kind)
		{
		case WindowKind::XYview:
			log_scale_slice_xy_enabled = value;
			break;
		case WindowKind::XZview:
			log_scale_slice_xz_enabled = value;
			break;
		case WindowKind::YZview:
			log_scale_slice_yz_enabled = value;
			break;
		}
	}

	void ComputeDescriptor::set_accumulation(WindowKind kind, bool value)
	{
		switch (kind)
		{
		case WindowKind::XYview:
			img_acc_slice_xy_enabled = value;
			break;
		case WindowKind::XZview:
			img_acc_slice_xz_enabled = value;
			break;
		case WindowKind::YZview:
			img_acc_slice_yz_enabled = value;
			break;
		}
	}

	void ComputeDescriptor::set_accumulation_level(WindowKind kind, float value)
	{
		switch (kind)
		{
		case WindowKind::XYview:
			img_acc_slice_xy_level = value;
			break;
		case WindowKind::XZview:
			img_acc_slice_xz_level = value;
			break;
		case WindowKind::YZview:
			img_acc_slice_yz_level = value;
			break;
		}
	}
}
