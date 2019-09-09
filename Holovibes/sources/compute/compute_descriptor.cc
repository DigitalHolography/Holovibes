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
		algorithm(Algorithm::None),
		compute_mode(Computation::Stop),
		nSize(1),
		pindex(0),
		lambda(532e-9f),
		interpolation_enabled(false),
		interp_lambda(532e-9f),
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
		divide_convolution_enabled(false),
		croped_stft(false),
		filter_2d_enabled(false),
		average_enabled(false),
		contrast_min_slice_xy(1.f),
		contrast_max_slice_xy(65535.f),
		contrast_min_slice_xz(1.f),
		contrast_min_slice_yz(1.f),
		contrast_max_slice_xz(65535.f),
		contrast_max_slice_yz(65535.f),
		scale_bar_correction_factor(1),
		vibrometry_q(0),
		autofocus_size(10),
		convo_matrix_width(0),
		convo_matrix_height(0),
		convo_matrix_z(0),
		autofocus_z_min(0.f),
		autofocus_z_max(1.f),
		autofocus_z_div(10),
		autofocus_z_iter(5),
		is_cine_file(false),
		pixel_size(5.42f),
		img_acc_slice_xy_enabled(false),
		img_acc_slice_xz_enabled(false),
		img_acc_slice_yz_enabled(false),
		img_acc_slice_xy_level(1),
		img_acc_slice_xz_level(1),
		img_acc_slice_yz_level(1),
		p_accu_enabled(false),
		p_acc_level(1),
		x_accu_enabled(false),
		x_acc_level(1),
		y_accu_enabled(false),
		y_acc_level(1),
		stft_level(16),
		stft_steps(1),
		ref_diff_level(15),
		ref_diff_enabled(false),
		ref_sliding_enabled(false),
		stft_view_enabled(false),
		current_window(WindowKind::XYview),
		cuts_contrast_p_offset(2),
		display_rate(30),
		xy_stabilization_enabled(false),
		xy_stabilization_paused(false),
		xy_stabilization_show_convolution(false),
		composite_p_red(0),
		composite_p_blue(0),
		weight_r(1),
		weight_g(1),
		weight_b(1)
	{

	}

	ComputeDescriptor::~ComputeDescriptor()
	{

	}
	
	ComputeDescriptor& ComputeDescriptor::operator=(const ComputeDescriptor& cd)
	{
		algorithm = cd.algorithm.load();
		compute_mode = cd.compute_mode.load();
		nSize = cd.nSize.load();
		pindex = cd.pindex.load();
		lambda = cd.lambda.load();
		zdistance = cd.zdistance.load();
		img_type = cd.img_type.load();
		unwrap_history_size = cd.unwrap_history_size.load();
		special_buffer_size = cd.special_buffer_size.load();
		log_scale_slice_xy_enabled = cd.log_scale_slice_xy_enabled.load();
		log_scale_slice_xz_enabled = cd.log_scale_slice_xz_enabled.load();
		log_scale_slice_yz_enabled = cd.log_scale_slice_yz_enabled.load();
		shift_corners_enabled = cd.shift_corners_enabled.load();
		contrast_enabled = cd.contrast_enabled.load();
		vibrometry_enabled = cd.vibrometry_enabled.load();
		convolution_enabled = cd.convolution_enabled.load();
		filter_2d_enabled = cd.filter_2d_enabled.load();
		average_enabled = cd.average_enabled.load();
		contrast_min_slice_xy = cd.contrast_min_slice_xy.load();
		contrast_max_slice_xy = cd.contrast_max_slice_xy.load();
		contrast_min_slice_xz = cd.contrast_min_slice_xz.load();
		contrast_min_slice_yz = cd.contrast_min_slice_yz.load();
		contrast_max_slice_xz = cd.contrast_max_slice_xz.load();
		contrast_max_slice_yz = cd.contrast_max_slice_yz.load();
		vibrometry_q = cd.vibrometry_q.load();
		autofocus_size = cd.autofocus_size.load();
		convo_matrix_width = cd.convo_matrix_width.load();
		convo_matrix_height = cd.convo_matrix_height.load();
		convo_matrix_z = cd.convo_matrix_z.load();
		autofocus_z_min = cd.autofocus_z_min.load();
		autofocus_z_max = cd.autofocus_z_max.load();
		autofocus_z_div = cd.autofocus_z_div.load();
		autofocus_z_iter = cd.autofocus_z_iter.load();
		is_cine_file = cd.is_cine_file.load();
		pixel_size = cd.pixel_size.load();
		img_acc_slice_xy_enabled = cd.img_acc_slice_xy_enabled.load();
		img_acc_slice_xz_enabled = cd.img_acc_slice_xz_enabled.load();
		img_acc_slice_yz_enabled = cd.img_acc_slice_yz_enabled.load();
		img_acc_slice_xy_level = cd.img_acc_slice_xy_level.load();
		img_acc_slice_xz_level = cd.img_acc_slice_xz_level.load();
		img_acc_slice_yz_level = cd.img_acc_slice_yz_level.load();
		stft_level = cd.stft_level.load();
		stft_steps = cd.stft_steps.load();
		ref_diff_level = cd.ref_diff_level.load();
		ref_diff_enabled = cd.ref_diff_enabled.load();
		ref_sliding_enabled = cd.ref_sliding_enabled.load();
		stft_view_enabled = cd.stft_view_enabled.load();
		current_window = cd.current_window.load();
		cuts_contrast_p_offset = cd.cuts_contrast_p_offset.load();
		display_rate = cd.display_rate.load();
		stft_slice_cursor = cd.stft_slice_cursor;
		signal_zone = cd.signal_zone;
		noise_zone = cd.noise_zone;
		autofocus_zone = cd.autofocus_zone;
		stft_roi_zone = cd.stft_roi_zone;
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

	void ComputeDescriptor::autofocusZone(units::RectFd& rect, AccessMode m)
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

	units::RectFd ComputeDescriptor::getStabilizationZone() const
	{
		LockGuard g(mutex_);
		return stabilization_zone;
	}

	void ComputeDescriptor::setStabilizationZone(const units::RectFd& rect)
	{
		LockGuard g(mutex_);
		stabilization_zone = rect;
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
}