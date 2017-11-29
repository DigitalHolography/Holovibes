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
#include "stft.cuh"

using holovibes::compute::FourierTransform;
using holovibes::compute::Autofocus;
using holovibes::Queue;
using holovibes::FnVector;

FourierTransform::FourierTransform(FnVector& fn_vect,
	cuComplex* const& gpu_input_buffer,
	const Autofocus& autofocus,
	const camera::FrameDescriptor& fd,
	const holovibes::ComputeDescriptor& cd,
	const cufftHandle& plan2d,
	const cufftHandle& plan1d_stft)
	: gpu_lens_(nullptr)
	, gpu_filter2d_buffer_(nullptr)
	, fn_vect_(fn_vect)
	, gpu_input_buffer_(gpu_input_buffer)
	, autofocus_(autofocus)
	, fd_(fd)
	, cd_(cd)
	, plan2d_(plan2d)
	, plan1d_stft_(plan1d_stft)
{
	gpu_lens_.resize(fd_.frame_res());
	gpu_filter2d_buffer_.resize(fd_.frame_res());
}

void FourierTransform::insert_fft()
{
	filter2d_zone_ = cd_.getStftZone();
	if (cd_.filter_2d_enabled)
		insert_filter2d();

	// Applying fresnel transform only when filter2d isn't in filtering mode (when overlay isn't released yet)
	if (!cd_.filter_2d_enabled || filter2d_zone_.area())
	{
		if (cd_.algorithm == Algorithm::FFT1)
			insert_fft1();
		else if (cd_.algorithm == Algorithm::FFT2)
			insert_fft2();
		fn_vect_.push_back([=]() {
			enqueue_lens(gpu_lens_queue_.get(), gpu_lens_.get(), fd_);
		});
	}
	//insert_stft();
}

void FourierTransform::insert_filter2d()
{
	fn_vect_.push_back(std::bind(
		filter2D,
		gpu_input_buffer_,
		gpu_filter2d_buffer_.get(),
		plan2d_,
		filter2d_zone_,
		fd_,
		static_cast<cudaStream_t>(0)));
}

void FourierTransform::insert_fft1()
{
	const float z = autofocus_.get_zvalue();
	fft1_lens(
		gpu_lens_.get(),
		fd_,
		cd_.lambda.load(),
		z,
		cd_.pixel_size);

	fn_vect_.push_back(std::bind(
		fft_1,
		gpu_input_buffer_,
		gpu_lens_.get(),
		plan2d_,
		fd_.frame_res(),
		static_cast<cudaStream_t>(0)));
}

void FourierTransform::insert_fft2()
{
	const float z = autofocus_.get_zvalue();
	fft2_lens(
		gpu_lens_.get(),
		fd_,
		cd_.lambda.load(),
		z,
		cd_.pixel_size);

	fn_vect_.push_back(std::bind(
		fft_2,
		gpu_input_buffer_,
		gpu_lens_.get(),
		plan2d_,
		fd_,
		static_cast<cudaStream_t>(0)));
}

/*void FourierTransform::insert_stft()
{
	fn_vect_.push_back(std::bind(
		queue_enqueue,
		this,
		gpu_input_buffer_,
		gpu_stft_queue_));

	fn_vect_.push_back(std::bind(
		stft_handler,
		this,
		gpu_input_buffer_,
		static_cast<cufftComplex *>(gpu_stft_queue_->get_buffer())));
}*/

Queue *FourierTransform::get_lens_queue()
{
	if (!gpu_lens_queue_ && cd_.gpu_lens_display_enabled)
	{
		auto fd = fd_;
		fd.depth = 8;
		gpu_lens_queue_ = std::make_unique<Queue>(fd, 16, "GPU lens queue");
	}
	return gpu_lens_queue_.get();
}

void FourierTransform::enqueue_lens(Queue *queue, cuComplex *lens_buffer, const camera::FrameDescriptor& input_fd)
{
	if (queue)
	{
		cuComplex* copied_lens_ptr = static_cast<cuComplex*>(queue->get_end());
		queue->enqueue(lens_buffer);
		normalize_complex(copied_lens_ptr, input_fd.frame_res());
	}
}

/*void FourierTransform::stft_handler(cufftComplex* input, cufftComplex* output)
{
	static ushort mouse_posx;
	static ushort mouse_posy;

	stft_frame_counter--;
	bool b = false;
	if (stft_frame_counter == 0)
	{
		b = true;
		stft_frame_counter = cd_.stft_steps;
	}
	std::lock_guard<std::mutex> Guard(stftGuard);

	if (!cd_.vibrometry_enabled)
	{
		stft(input,
			output,
			gpu_stft_buffer_,
			plan1d_stft_,
			cd_.nsamples,
			cd_.pindex,
			cd_.pindex,
			cd_.nsamples,
			fd_.width,
			fd_.height,
			b,
			cd_.croped_stft,
			cd_.getZoomedZone(),
			gpu_cropped_stft_buf_.get(),
			static_cast<cudaStream_t>(0));
	}
	else
	{
		stft(
			input,
			static_cast<cufftComplex *>(gpu_stft_queue_->get_buffer()),
			gpu_stft_buffer_,
			plan1d_stft_,
			cd_.nsamples,
			cd_.pindex,
			cd_.vibrometry_q,
			cd_.nsamples,
			fd_.width,
			fd_.height,
			b,
			cd_.croped_stft,
			cd_.getZoomedZone(),
			gpu_cropped_stft_buf_.get(),
			static_cast<cudaStream_t>(0));
	}
	if (cd_.stft_view_enabled && b)
	{
		// Conservation of the coordinates when cursor is outside of the window
		units::PointFd cursorPos;
		cd_.stftCursor(cursorPos, AccessMode::Get);
		const ushort width = fd_.width;
		const ushort height = fd_.height;
		if (static_cast<ushort>(cursorPos.x()) < width &&
			static_cast<ushort>(cursorPos.y()) < height)
		{
			mouse_posx = cursorPos.x();
			mouse_posy = cursorPos.y();
		}
		// -----------------------------------------------------
		stft_view_begin(gpu_stft_buffer_,
			gpu_float_cut_xz_,
			gpu_float_cut_yz_,
			cd_.x_accu_enabled ? cd_.x_accu_min_level : mouse_posx,
			cd_.y_accu_enabled ? cd_.y_accu_min_level : mouse_posy,
			cd_.x_accu_enabled ? cd_.x_accu_max_level : mouse_posx,
			cd_.y_accu_enabled ? cd_.y_accu_max_level : mouse_posy,
			width,
			height,
			cd_.img_type,
			cd_.nsamples,
			cd_.img_acc_slice_xz_enabled ? cd_.img_acc_slice_xz_level : 1,
			cd_.img_acc_slice_yz_enabled ? cd_.img_acc_slice_yz_level : 1,
			cd_.img_type);
	}
	stft_handle = true;
} */
