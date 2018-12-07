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

#include "fourier_transform.hh"
#include "compute_descriptor.hh"
#include "tools_conversion.cuh"
#include "filter2d.cuh"
#include "fft1.cuh"
#include "fft2.cuh"
#include "transforms.cuh"
#include "stft.cuh"
#include "icompute.hh"
#include "info_manager.hh"
#include "debug_img.cuh"

using holovibes::compute::FourierTransform;
using holovibes::compute::Autofocus;
using holovibes::Queue;
using holovibes::FnVector;

FourierTransform::FourierTransform(FnVector& fn_vect,
	const holovibes::CoreBuffers& buffers,
	const std::unique_ptr<Autofocus>& autofocus,
	const camera::FrameDescriptor& fd,
	holovibes::ComputeDescriptor& cd,
	const cufftHandle& plan2d,
	holovibes::Stft_env& stft_env,
	holovibes::Stft_env& stft_longtimes_env)
	: gpu_lens_()
	, gpu_lens_queue_()
	, gpu_filter2d_buffer_()
	, gpu_cropped_stft_buf_()
	, gpu_cropped_stft_longtimes_buf_()
	, fn_vect_(fn_vect)
	, buffers_(buffers)
	, autofocus_(autofocus)
	, fd_(fd)
	, cd_(cd)
	, plan2d_(plan2d)
	, stft_env_(stft_env)
	, stft_longtimes_env_(stft_longtimes_env)
{
	gpu_lens_.resize(fd_.frame_res());
	gpu_filter2d_buffer_.resize(fd_.frame_res());

	std::stringstream ss;
	ss << "(X1,Y1,X2,Y2) = (";
	if (cd_.croped_stft)
	{
		auto zone = cd_.getZoomedZone();
		gpu_cropped_stft_buf_.resize(zone.area() * cd_.nSize);
		//gpu_cropped_stft_longtimes_buf_.resize(zone.area() * cd_.nSize_longtimes);
		ss << zone.x() << "," << zone.y() << "," << zone.right() << "," << zone.bottom() << ")";
	}
	else
		ss << "0,0," << fd_.width - 1 << "," << fd_.height - 1 << ")";

	gui::InfoManager::get_manager()->insert_info(gui::InfoManager::STFT_ZONE, "STFT Zone", ss.str());
}


void FourierTransform::allocate_filter2d(unsigned int n, bool is_longtimes)
{
	if (!is_longtimes)
	{
		if (cd_.croped_stft)
			gpu_cropped_stft_buf_.resize(cd_.getZoomedZone().area() * n);
		else
			gpu_cropped_stft_buf_.reset();
	}
	/*else
	{
		if (cd_.croped_stft)
			gpu_cropped_stft_longtimes_buf_.resize(cd_.getZoomedZone().area() * n);
		else
			gpu_cropped_stft_longtimes_buf_.reset();
	}*/
}

void FourierTransform::insert_fft()
{
	filter2d_zone_ = cd_.getStftZone();
	if (cd_.filter_2d_enabled)
		insert_filter2d();

	// In filter 2D: Applying fresnel transform only when filter2d overlay is release
	if (!cd_.filter_2d_enabled || filter2d_zone_.area())
	{
		if (cd_.algorithm == Algorithm::FFT1)
			insert_fft1();
		else if (cd_.algorithm == Algorithm::FFT2)
			insert_fft2();
		if (cd_.algorithm == Algorithm::FFT1 || cd_.algorithm == Algorithm::FFT2)
			fn_vect_.push_back([=]() {
				enqueue_lens();
			});
	}
}

void FourierTransform::insert_filter2d()
{
	fn_vect_.push_back([=]() {
		filter2D(
			buffers_.gpu_input_buffer_,
			gpu_filter2d_buffer_,
			plan2d_,
			filter2d_zone_,
			fd_);
	});
}

void FourierTransform::insert_fft1()
{
	const float z = autofocus_->get_zvalue();
	fft1_lens(gpu_lens_,
		fd_,
		cd_.lambda,
		z,
		cd_.pixel_size);

	compute_zernike(z);

	fn_vect_.push_back([=]() {
		fft_1(
			buffers_.gpu_input_buffer_,
			gpu_lens_.get(),
			plan2d_,
			fd_.frame_res());
	});
}

void FourierTransform::insert_fft2()
{
	const float z = autofocus_->get_zvalue();
	fft2_lens(gpu_lens_,
		fd_,
		cd_.lambda,
		z,
		cd_.pixel_size);

	compute_zernike(z);

	fn_vect_.push_back([=]() {
		fft_2(
			buffers_.gpu_input_buffer_,
			gpu_lens_,
			plan2d_,
			fd_);
	});
}

void FourierTransform::insert_stft()
{
	fn_vect_.push_back([=]() { queue_enqueue(buffers_.gpu_input_buffer_, stft_env_.gpu_stft_queue_.get()); });

	fn_vect_.push_back([=]() { stft_handler(); });
}

void FourierTransform::insert_stft_longtimes()
{
	//TODO ELLENA 
	fn_vect_.push_back([=]() { queue_enqueue(stft_env_.gpu_stft_buffer_.get() + cd_.pindex * fd_.frame_res(), stft_longtimes_env_.gpu_stft_queue_.get()); });

	fn_vect_.push_back([=]() { stft_longtimes_handler(); });
}


std::unique_ptr<Queue>& FourierTransform::get_lens_queue()
{
	if (!gpu_lens_queue_ && cd_.gpu_lens_display_enabled)
	{
		auto fd = fd_;
		fd.depth = 8;
		gpu_lens_queue_ = std::make_unique<Queue>(fd, 16, "GPU lens queue");
	}
	return gpu_lens_queue_;
}

void FourierTransform::enqueue_lens()
{
	if (gpu_lens_queue_)
	{
		// Getting the pointer in the location of the next enqueued element
		cuComplex* copied_lens_ptr = static_cast<cuComplex*>(gpu_lens_queue_->get_end());
		gpu_lens_queue_->enqueue(gpu_lens_);
		// Normalizing the newly enqueued element
		normalize_complex(copied_lens_ptr, fd_.frame_res());
	}
}

void FourierTransform::stft_handler()
{
	static ushort mouse_posx;
	static ushort mouse_posy;

	stft_env_.stft_frame_counter_--;
	bool b = false;
	if (stft_env_.stft_frame_counter_ == 0)
	{
		b = true;
		stft_env_.stft_frame_counter_ = cd_.stft_steps;
	}
	std::lock_guard<std::mutex> Guard(stft_env_.stftGuard_);

	if (!cd_.vibrometry_enabled)
		//reuse for stft longtimes
		stft(buffers_.gpu_input_buffer_,
			stft_env_.gpu_stft_queue_.get(),
			stft_env_.gpu_stft_buffer_,
			stft_env_.plan1d_stft_,
			cd_.pindex,
			fd_.width,
			fd_.height,
			b,
			cd_,
			gpu_cropped_stft_buf_,
			false);
	else
	{
		stft(
			buffers_.gpu_input_buffer_,
			stft_env_.gpu_stft_queue_.get(),
			stft_env_.gpu_stft_buffer_,
			stft_env_.plan1d_stft_,
			cd_.vibrometry_q,
			fd_.width,
			fd_.height,
			b,
			cd_,
			gpu_cropped_stft_buf_,
			false);
	}
	if (cd_.stft_view_enabled && b)
	{
		// Conservation of the coordinates when cursor is outside of the window
		units::PointFd cursorPos = cd_.getStftCursor();
		const ushort width = fd_.width;
		const ushort height = fd_.height;
		if (static_cast<ushort>(cursorPos.x()) < width &&
			static_cast<ushort>(cursorPos.y()) < height)
		{
			mouse_posx = cursorPos.x();
			mouse_posy = cursorPos.y();
		}
		// -----------------------------------------------------
		stft_view_begin(stft_env_.gpu_stft_buffer_,
			buffers_.gpu_float_cut_xz_,
			buffers_.gpu_float_cut_yz_,
			mouse_posx,
			mouse_posy,
			mouse_posx + (cd_.x_accu_enabled ? cd_.x_acc_level.load() : 0),
			mouse_posy + (cd_.y_accu_enabled ? cd_.y_acc_level.load() : 0),
			width,
			height,
			cd_.img_type,
			cd_.nSize,
			cd_.img_acc_slice_xz_enabled ? cd_.img_acc_slice_xz_level.load() : 1,
			cd_.img_acc_slice_yz_enabled ? cd_.img_acc_slice_yz_level.load() : 1,
			cd_.img_type);
	}
	stft_env_.stft_handle_ = true;
}

void FourierTransform::stft_longtimes_handler() //TODO ELLENA
{
	static ushort mouse_posx;
	static ushort mouse_posy;

	stft_longtimes_env_.stft_frame_counter_--;
	bool b = false;
	if (stft_longtimes_env_.stft_frame_counter_ == 0)
	{
		b = true;
		stft_longtimes_env_.stft_frame_counter_ = cd_.stft_longtimes_steps;
	}
	std::lock_guard<std::mutex> Guard(stft_longtimes_env_.stftGuard_);

	if (!cd_.vibrometry_enabled)
		stft(buffers_.gpu_input_buffer_,
			stft_longtimes_env_.gpu_stft_queue_.get(),
			stft_longtimes_env_.gpu_stft_buffer_,
			stft_longtimes_env_.plan1d_stft_,
			cd_.pindex_longtimes,
			fd_.width,
			fd_.height,
			b,
			cd_,
			gpu_cropped_stft_longtimes_buf_,
			true);
	else
	{
		stft(
			buffers_.gpu_input_buffer_,
			stft_longtimes_env_.gpu_stft_queue_.get(),
			stft_longtimes_env_.gpu_stft_buffer_,
			stft_longtimes_env_.plan1d_stft_,
			cd_.vibrometry_q,
			fd_.width,
			fd_.height,
			b,
			cd_,
			gpu_cropped_stft_longtimes_buf_,
			true);
	}


	from_gpu_img_to_csv((float*)(void*)stft_longtimes_env_.gpu_stft_queue_.get()->get_buffer(), fd_.width * fd_.height * 2, "part2-stft_longtimes_env_.gpu_stft_queue_.csv");
	from_gpu_img_to_csv((float*)(void*)stft_longtimes_env_.gpu_stft_buffer_.get(), fd_.width * fd_.height * 2, "part2-stft_longtimes_env_.gpu_stft_buffer_.csv");

	if (cd_.stft_view_enabled && b)
	{
		// Conservation of the coordinates when cursor is outside of the window
		units::PointFd cursorPos = cd_.getStftCursor();
		const ushort width = fd_.width;
		const ushort height = fd_.height;
		if (static_cast<ushort>(cursorPos.x()) < width &&
			static_cast<ushort>(cursorPos.y()) < height)
		{
			mouse_posx = cursorPos.x();
			mouse_posy = cursorPos.y();
		}
		// -----------------------------------------------------
		stft_view_begin(stft_longtimes_env_.gpu_stft_buffer_,
			buffers_.gpu_float_cut_xz_,
			buffers_.gpu_float_cut_yz_,
			mouse_posx,
			mouse_posy,
			mouse_posx + (cd_.x_accu_enabled ? cd_.x_acc_level.load() : 0),
			mouse_posy + (cd_.y_accu_enabled ? cd_.y_acc_level.load() : 0),
			width,
			height,
			cd_.img_type,
			cd_.nSize_longtimes,
			cd_.img_acc_slice_xz_enabled ? cd_.img_acc_slice_xz_level.load() : 1,
			cd_.img_acc_slice_yz_enabled ? cd_.img_acc_slice_yz_level.load() : 1,
			cd_.img_type);
	}
	stft_longtimes_env_.stft_handle_ = true;
}

void FourierTransform::compute_zernike(const float z)
{
	if (cd_.zernike_enabled && cd_.zernike_m <= cd_.zernike_n)
		zernike_lens(gpu_lens_,
			fd_,
			cd_.lambda,
			z,
			cd_.pixel_size,
			cd_.zernike_m,
			cd_.zernike_n,
			cd_.zernike_factor);
}
