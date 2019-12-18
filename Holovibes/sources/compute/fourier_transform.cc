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
#include "tools_conversion.cuh"
#include "tools_compute.cuh"
#include "filter2d.cuh"
#include "fft1.cuh"
#include "fft2.cuh"
#include "transforms.cuh"
#include "stft.cuh"
#include "icompute.hh"
#include "info_manager.hh"
#include "debug_img.cuh"

#include "cublas_handle.hh"
#include "cusolver_handle.hh"
#include "nppi_functions.hh"

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
	holovibes::Stft_env& stft_env)
	: gpu_lens_()
	, gpu_lens_queue_()
	, gpu_filter2d_buffer_()
	, gpu_cropped_stft_buf_()
	, fn_vect_(fn_vect)
	, buffers_(buffers)
	, autofocus_(autofocus)
	, fd_(fd)
	, cd_(cd)
	, plan2d_(plan2d)
	, stft_env_(stft_env)
{
	gpu_lens_.resize(fd_.frame_res());
	gpu_filter2d_buffer_.resize(fd_.frame_res());

	std::stringstream ss;
	ss << "(X1,Y1,X2,Y2) = (";
	if (cd_.croped_stft)
	{
		auto zone = cd_.getZoomedZone();
		gpu_cropped_stft_buf_.resize(zone.area() * cd_.nSize);
		ss << zone.x() << "," << zone.y() << "," << zone.right() << "," << zone.bottom() << ")";
	}
	else
		ss << "0,0," << fd_.width - 1 << "," << fd_.height - 1 << ")";

	gui::InfoManager::get_manager()->insert_info(gui::InfoManager::STFT_ZONE, "STFT Zone", ss.str());
}


void FourierTransform::allocate_filter2d(unsigned int n)
{
	if (cd_.croped_stft)
		gpu_cropped_stft_buf_.resize(cd_.getZoomedZone().area() * n);
	else
		gpu_cropped_stft_buf_.reset();
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
	if (cd_.filter_2d_type == Filter2DType::BandPass)
	{
		filter2d_subzone_ = cd_.getFilter2DSubZone();
		fn_vect_.push_back([=](){
			filter2D_BandPass(
				buffers_.gpu_input_buffer_,
				gpu_filter2d_buffer_,
				plan2d_,
				filter2d_zone_,
				filter2d_subzone_,
				fd_
			);
		});
	}
	else//Low pass or High pass
	{
		bool exclude_roi = cd_.filter_2d_type == Filter2DType::HighPass;
		fn_vect_.push_back([=]() {
			filter2D(
				buffers_.gpu_input_buffer_,
				gpu_filter2d_buffer_,
				plan2d_,
				filter2d_zone_,
				fd_,
				exclude_roi);
		});
	}
	
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

// Useful to debug ``insert_eigenvalue_filter()``
static void print_matrix(const char* name, cuComplex* matrix, unsigned rows, unsigned columns, bool column_major)
{
	std::cout << name << ": " << std::endl;
	cuComplex* tmp = new cuComplex[rows * columns];
	cudaMemcpy(tmp, matrix, rows * columns * sizeof(cuComplex), cudaMemcpyDeviceToHost);
	for (unsigned y = 0; y < rows; ++y)
	{
		for (unsigned x = 0; x < columns; ++x)
		{
			if (column_major)
			{
				const cuComplex& val = tmp[x * rows + y];
				std::printf("%f+%fi ", val.x, val.y);
			}
			else
			{
				const cuComplex& val = tmp[y * columns + x];
				std::printf("%f+%fi ", val.x, val.y);
			}
		}
		std::cout << ";" << std::endl;
	}
	std::cout << std::endl;
	delete[] tmp;
}

void FourierTransform::insert_eigenvalue_filter()
{
	fn_vect_.push_back([=]() { queue_enqueue(buffers_.gpu_input_buffer_, stft_env_.gpu_stft_queue_.get()); });

	fn_vect_.push_back([=]() {
		bool b = false;
		stft_env_.stft_frame_counter_--;
		if (stft_env_.stft_frame_counter_ == 0)
		{
			b = true;
			stft_env_.stft_frame_counter_ = cd_.stft_steps;
		}

		unsigned short p_acc = cd_.p_acc_level + 1;
		unsigned short p = cd_.pindex;
		if (p + p_acc > cd_.nSize)
		{
			p_acc = cd_.nSize - p;
		}

		if (b)
		{
			cudaError_t cuda_status;
			cublasStatus_t cublas_status;
			cusolverStatus_t cusolver_status;

			cuComplex alpha{ 1, 0 };
			cuComplex beta{ 0, 0 };

			cudaMemcpy(stft_env_.gpu_stft_buffer_.get(), stft_env_.gpu_stft_queue_->get_buffer(), fd_.frame_res() * cd_.nSize * sizeof(cuComplex), cudaMemcpyDeviceToDevice);

			cuComplex* H = stft_env_.gpu_stft_buffer_.get();
			cuComplex* cov = stft_env_.svd_cov.get();

			// cov = H' * H
			cublas_status = cublasCgemm(cuda_tools::CublasHandle::instance(),
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
				cd_.nSize);
			cuda_status = cudaDeviceSynchronize();
			assert(cuda_status == cudaSuccess);
			assert(cublas_status == CUBLAS_STATUS_SUCCESS && "cov = H' * H failed");

			// Setup eigen values parameters
			float* W = stft_env_.svd_eigen_values.get();
			int lwork = 0;
			cusolver_status = cusolverDnCheevd_bufferSize(cuda_tools::CusolverHandle::instance(),
				CUSOLVER_EIG_MODE_VECTOR,
				CUBLAS_FILL_MODE_LOWER,
				cd_.nSize,
				cov,
				cd_.nSize,
				W,
				&lwork);
			assert(cusolver_status == CUSOLVER_STATUS_SUCCESS && "Could not allocate work buffer");
			cuda_tools::UniquePtr<cuComplex> work(lwork);

			// Find eigen values and eigen vectors of cov
			// W will contain sorted eigen values
			// cov will contain eigen vectors
			cusolver_status = cusolverDnCheevd(cuda_tools::CusolverHandle::instance(),
				CUSOLVER_EIG_MODE_VECTOR,
				CUBLAS_FILL_MODE_LOWER,
				cd_.nSize,
				cov,
				cd_.nSize,
				W,
				work.get(),
				lwork,
				stft_env_.svd_dev_info.get());
			cuda_status = cudaDeviceSynchronize();
			assert(cuda_status == cudaSuccess);
			assert(cusolver_status == CUSOLVER_STATUS_SUCCESS && "Could not find eigen values / vectors of cov");

			// eigen vectors
			cuComplex* V = cov;



			/* Filtering the eigenvector matrix according to p and p_acc
			   The matrix should look like this:

				 0  ... p-1  p ... p_acc p+p_acc+1 ... nSize
			   ------------------------------------------
			   | 0  ...  0   X ...   X     0     ...  0 |
			   | 0  ...  0   X ...   X     0     ...  0 |
			   | 0  ...  0   X ...   X     0     ...  0 |
			   | 0  ...  0   X ...   X     0     ...  0 |
			   | 0  ...  0   X ...   X     0     ...  0 |
			   ------------------------------------------ */

			cudaMemset(V, 0, p * cd_.nSize * sizeof(cuComplex));
			cudaMemset(V + p * cd_.nSize + p_acc * cd_.nSize, 0, cd_.nSize * (cd_.nSize - (p + p_acc)) * sizeof(cuComplex));

			cuComplex* tmp = stft_env_.svd_tmp_buffer.get();

			// tmp = V * V'
			cublas_status = cublasCgemm(cuda_tools::CublasHandle::instance(),
				CUBLAS_OP_N,
				CUBLAS_OP_C,
				cd_.nSize,
				cd_.nSize,
				cd_.nSize,
				&alpha,
				V,
				cd_.nSize,
				V,
				cd_.nSize,
				&beta,
				tmp,
				cd_.nSize);
			cuda_status = cudaDeviceSynchronize();
			assert(cuda_status == cudaSuccess);
			assert(cublas_status == CUBLAS_STATUS_SUCCESS && "tmp = V * V' failed");

			cuComplex* H_noise = stft_env_.svd_noise.get();

			// H_noise = H * tmp
			cublas_status = cublasCgemm(cuda_tools::CublasHandle::instance(),
				CUBLAS_OP_N,
				CUBLAS_OP_N,
				fd_.frame_res(),
				cd_.nSize,
				cd_.nSize,
				&alpha,
				H,
				fd_.frame_res(),
				tmp,
				cd_.nSize,
				&beta,
				H_noise,
				fd_.frame_res());
			cuda_status = cudaDeviceSynchronize();
			assert(cuda_status == cudaSuccess);
			assert(cublas_status == CUBLAS_STATUS_SUCCESS && "H_noise = H * tmp failed");

			subtract_frame_complex(H, H_noise, H, fd_.frame_res() * cd_.nSize);
			// cudaMemcpy(H, H_noise.get(), fd_.frame_res() * cd_.nSize * sizeof(cuComplex), cudaMemcpyDeviceToDevice);
		}
		average_complex_images(stft_env_.gpu_stft_buffer_.get(), buffers_.gpu_input_buffer_.get(), fd_.frame_res(), cd_.nSize);
	});
}