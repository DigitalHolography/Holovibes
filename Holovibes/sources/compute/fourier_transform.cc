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

#include <sstream>

#include "fourier_transform.hh"
#include "compute_descriptor.hh"
#include "cublas_handle.hh"
#include "cusolver_handle.hh"
#include "nppi_functions.hh"
#include "info_manager.hh"
#include "icompute.hh"

#include "tools_conversion.cuh"
#include "tools_compute.cuh"
#include "filter2d.cuh"
#include "fft1.cuh"
#include "fft2.cuh"
#include "transforms.cuh"
#include "stft.cuh"
#include "cuda_tools/cufft_handle.hh"
#include "cuda_memory.cuh"


using holovibes::compute::FourierTransform;
using holovibes::Queue;
using holovibes::FnVector;

FourierTransform::FourierTransform(FnVector& fn_vect,
	const holovibes::CoreBuffers& buffers,
	const camera::FrameDescriptor& fd,
	holovibes::ComputeDescriptor& cd,
	holovibes::cuda_tools::CufftHandle& plan2d,
	holovibes::Stft_env& stft_env)
	: gpu_lens_()
	, gpu_lens_queue_()
	, gpu_filter2d_buffer_()
	, fn_vect_(fn_vect)
	, buffers_(buffers)
	, fd_(fd)
	, cd_(cd)
	, plan2d_(plan2d)
	, stft_env_(stft_env)
{
	gpu_lens_.resize(fd_.frame_res());
	gpu_filter2d_buffer_.resize(fd_.frame_res() * cd_.stft_steps);

	std::stringstream ss;
	ss << "(X1,Y1,X2,Y2) = (" << "0,0," << fd_.width - 1 << "," << fd_.height - 1 << ")";

	gui::InfoManager::get_manager()->insert_info(gui::InfoManager::STFT_ZONE, "STFT Zone", ss.str());
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

	fn_vect_.push_back([=]() {
		stft_env_.gpu_stft_queue_->enqueue_multiple(buffers_.gpu_input_buffer_.get(), cd_.stft_steps);
	});
}

void FourierTransform::insert_filter2d()
{
	if (cd_.filter_2d_type == Filter2DType::BandPass)
	{
		filter2d_subzone_ = cd_.getFilter2DSubZone();
		fn_vect_.push_back([=](){
			filter2D_BandPass(
				buffers_.gpu_input_buffer_,
				gpu_filter2d_buffer_,
				cd_.stft_steps,
				plan2d_,
				filter2d_zone_,
				filter2d_subzone_,
				fd_
			);
		});
	}
	else //Low pass or High pass
	{
		bool exclude_roi = cd_.filter_2d_type == Filter2DType::HighPass;
		fn_vect_.push_back([=]() {
			filter2D(
				buffers_.gpu_input_buffer_,
				gpu_filter2d_buffer_,
				cd_.stft_steps,
				plan2d_,
				filter2d_zone_,
				fd_,
				exclude_roi);
		});
	}
}

void FourierTransform::insert_fft1()
{
	const float z = cd_.zdistance;

	fft1_lens(gpu_lens_,
		fd_,
		cd_.lambda,
		z,
		cd_.pixel_size);

	fn_vect_.push_back([=]() {
		fft_1(
			buffers_.gpu_input_buffer_,
			buffers_.gpu_input_buffer_,
			cd_.stft_steps,
			gpu_lens_.get(),
			plan2d_,
			fd_.frame_res());
	});
}

void FourierTransform::insert_fft2()
{
	const float z = cd_.zdistance;

	fft2_lens(gpu_lens_,
		fd_,
		cd_.lambda,
		z,
		cd_.pixel_size);

	fn_vect_.push_back([=]() {
		fft_2(
			buffers_.gpu_input_buffer_,
			buffers_.gpu_input_buffer_,
			cd_.stft_steps,
			gpu_lens_.get(),
			plan2d_,
			fd_);
	});
}

void FourierTransform::insert_store_p_frame()
{
	fn_vect_.push_back([=]() {
		const int frame_res = fd_.frame_res();

		/* Copies with DeviceToDevice (which is the case here) are asynchronous with respect to the host
		* but never overlap with kernel execution*/
		cudaXMemcpyAsync(stft_env_.gpu_p_frame_,
				(cuComplex*)stft_env_.gpu_stft_buffer_ + cd_.pindex * frame_res,
				sizeof(cuComplex) * frame_res,
				cudaMemcpyDeviceToDevice);
	});
}

void FourierTransform::insert_stft()
{
	fn_vect_.push_back([=]() {
		stft(stft_env_.gpu_stft_queue_.get(),
		stft_env_.gpu_stft_buffer_,
		stft_env_.plan1d_stft_);
	 });
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

void FourierTransform::insert_eigenvalue_filter()
{
	fn_vect_.push_back([=]() {
		unsigned short p_acc = cd_.p_accu_enabled ? cd_.p_acc_level + 1 : 1;
		unsigned short p = cd_.pindex;
		if (p + p_acc > cd_.nSize)
		{
			p_acc = cd_.nSize - p;
		}

		constexpr cuComplex alpha{ 1, 0 };
		constexpr cuComplex beta{ 0, 0 };

		cudaXMemcpy(stft_env_.gpu_stft_buffer_.get(),
					stft_env_.gpu_stft_queue_->get_data(),
					fd_.frame_res() * cd_.nSize * sizeof(cuComplex),
					cudaMemcpyDeviceToDevice);

		cuComplex* H = stft_env_.gpu_stft_buffer_.get();
		cuComplex* cov = stft_env_.svd_cov.get();

		// cov = H' * H
		cublasSafeCall(cublasCgemm(cuda_tools::CublasHandle::instance(),
			CUBLAS_OP_C,
			CUBLAS_OP_N,
			cd_.nSize,
			cd_.nSize,
			fd_.frame_res(),
			&alpha,
			H,
			fd_.frame_res(),
			H,
			fd_.frame_res(),
			&beta,
			cov,
			cd_.nSize));

		// Setup eigen values parameters
		float* W = stft_env_.svd_eigen_values.get();
		int lwork = 0;
		cusolverSafeCall(cusolverDnCheevd_bufferSize(cuda_tools::CusolverHandle::instance(),
			CUSOLVER_EIG_MODE_VECTOR,
			CUBLAS_FILL_MODE_LOWER,
			cd_.nSize,
			cov,
			cd_.nSize,
			W,
			&lwork));

		cuda_tools::UniquePtr<cuComplex> work(lwork);

		// Find eigen values and eigen vectors of cov
		// W will contain sorted eigen values
		// cov will contain eigen vectors
		cusolverSafeCall(cusolverDnCheevd(cuda_tools::CusolverHandle::instance(),
			CUSOLVER_EIG_MODE_VECTOR,
			CUBLAS_FILL_MODE_LOWER,
			cd_.nSize,
			cov,
			cd_.nSize,
			W,
			work.get(),
			lwork,
			stft_env_.svd_dev_info.get()));

		// eigen vectors
		cuComplex* V = cov;

		// H = H * V
		cublasSafeCall(cublasCgemm(cuda_tools::CublasHandle::instance(),
			CUBLAS_OP_N,
			CUBLAS_OP_N,
			fd_.frame_res(),
			cd_.nSize,
			cd_.nSize,
			&alpha,
			H,
			fd_.frame_res(),
			V,
			cd_.nSize,
			&beta,
			H,
			fd_.frame_res()));
	});
}

void FourierTransform::insert_stft_cuts_view()
{
	fn_vect_.push_back([=]() {
		if (cd_.stft_view_enabled)
		{
			static ushort mouse_posx;
			static ushort mouse_posy;

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
				buffers_.gpu_float_cut_xz_.get(),
				buffers_.gpu_float_cut_yz_.get(),
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
	});
}
