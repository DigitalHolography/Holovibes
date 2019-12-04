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

void print_matrix(const char* name, cuComplex* matrix, unsigned rows, unsigned columns, bool column_major)
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
	fn_vect_.push_back([=]() { 
		cuComplex val;
		val.x = 8000;
		val.y = -700;
		for (unsigned i = 0; i < fd_.frame_res(); ++i)
		{
			cudaMemcpy(buffers_.gpu_input_buffer_.get() + i, &val, sizeof(cuComplex), cudaMemcpyHostToDevice);
		}

		queue_enqueue(buffers_.gpu_input_buffer_, stft_env_.gpu_stft_queue_.get());
	});

	fn_vect_.push_back([=]() { 
		cudaError_t cuda_status;
		cublasStatus_t cublas_status;
		cusolverStatus_t cusolver_status;

		cuComplex alpha{ 1, 0 };
		cuComplex beta{ 0, 0 };

		// Transpose each image in gpu_stft_queue_ to have the right format for the multiplication
		for (unsigned i = 0; i < cd_.nSize; ++i)
		{
			cublas_status = cublasCgeam(cuda_tools::CublasHandle::instance(),
				CUBLAS_OP_T,
				CUBLAS_OP_N,
				fd_.height,
				fd_.width,
				&alpha,
				(const cuComplex*)stft_env_.gpu_stft_queue_->get_buffer() + i * fd_.frame_res(),
				fd_.height,
				&beta,
				stft_env_.gpu_stft_buffer_.get() + i * fd_.frame_res(),
				fd_.height,
				stft_env_.gpu_stft_buffer_.get() + i * fd_.frame_res(),
				fd_.height);
			cuda_status = cudaDeviceSynchronize();
			assert(cuda_status == cudaSuccess);
			assert(cublas_status == CUBLAS_STATUS_SUCCESS && "Could not transpose an image of gpu_stft_queue_");
		}

		cuComplex* H = stft_env_.gpu_stft_buffer_.get();
		cuda_tools::UniquePtr<cuComplex> cov(cd_.nSize * cd_.nSize);

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
			cov.get(),
			cd_.nSize);
		cuda_status = cudaDeviceSynchronize();
		assert(cuda_status == cudaSuccess);
		assert(cublas_status == CUBLAS_STATUS_SUCCESS && "cov = H' * H failed");

		print_matrix("cov", cov.get(), cd_.nSize, cd_.nSize, true);

		// Setup eigen values parameters
		cuda_tools::UniquePtr<float> W(cd_.nSize);
		int lwork = 0;
		cusolver_status = cusolverDnCheevd_bufferSize(cuda_tools::CusolverHandle::instance(), CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, cd_.nSize, cov.get(), cd_.nSize, W.get() , &lwork);
		assert(cusolver_status == CUSOLVER_STATUS_SUCCESS && "Could not allocate work buffer");
		cuda_tools::UniquePtr<cuComplex> work(lwork);
		cuda_tools::UniquePtr<int> dev_info(1);

		// Find eigen values and eigen vectors of cov
		// W will contain sorted eigen values
		// cov will contain eigen vectors
		cusolver_status = cusolverDnCheevd(cuda_tools::CusolverHandle::instance(),
			CUSOLVER_EIG_MODE_VECTOR,
			CUBLAS_FILL_MODE_LOWER,
			cd_.nSize,
			cov.get(),
			cd_.nSize,
			W.get(),
			work.get(),
			lwork,
			dev_info.get());
		cuda_status = cudaDeviceSynchronize();
		assert(cuda_status == cudaSuccess);
		assert(cusolver_status == CUSOLVER_STATUS_SUCCESS && "Could not find eigen values / vectors of cov");

		std::cout << "W: " << std::endl;
		float* print_eigenvalues = new float[cd_.nSize];
		cudaMemcpy(print_eigenvalues, W.get(), cd_.nSize * sizeof(float), cudaMemcpyDeviceToHost);
		for (unsigned y = 0; y < cd_.nSize; ++y)
		{
			for (unsigned x = 0; x < cd_.nSize; ++x)
			{
				if (x == y)
				{
					const float& val = print_eigenvalues[x];
					std::printf("%.30f ", val);
				}
				else
				{
					std::printf("0.000 ");
				}
			}
			std::cout << ";" << std::endl;
		}
		std::cout << std::endl;
		delete[] print_eigenvalues;

		// eigen vectors
		cuComplex* V = cov.get();

		// Filtering the eigenvector matrix according to p and p_acc
		/* cudaMemset(V, 0, (cd_.nSize * cd_.pindex + cd_.pindex) * sizeof(cuComplex));
		cuComplex* ptr = V + (cd_.nSize * cd_.pindex + cd_.pindex);
		for (unsigned i = 0; i < cd_.p_acc_level - 1; ++i)
		{
			cudaMemset(ptr + cd_.p_acc_level, 0, (cd_.nSize - cd_.p_acc_level) * sizeof(cuComplex));
			ptr += cd_.nSize;
		}
		cudaMemset(ptr + cd_.p_acc_level, 0, (cd_.nSize * (cd_.nSize - (cd_.pindex + cd_.p_acc_level)) + cd_.nSize - (cd_.pindex + cd_.p_acc_level)) * sizeof(cuComplex)); */

		print_matrix("V", V, cd_.nSize, cd_.nSize, true);
		
		cudaMemset(V, 0, cd_.pindex * cd_.nSize * sizeof(cuComplex));
		cudaMemset(V + cd_.pindex * cd_.nSize + cd_.p_acc_level * cd_.nSize, 0, cd_.nSize * (cd_.nSize - (cd_.pindex + cd_.p_acc_level)) * sizeof(cuComplex));

		print_matrix("V filtered", V, cd_.nSize, cd_.nSize, true);

		cuda_tools::UniquePtr<cuComplex> tmp(cd_.nSize * cd_.nSize);
		
		print_matrix("H", H, fd_.frame_res(), cd_.nSize, true);

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
			tmp.get(),
			cd_.nSize);
		cuda_status = cudaDeviceSynchronize();
		assert(cuda_status == cudaSuccess);
		assert(cublas_status == CUBLAS_STATUS_SUCCESS && "tmp = V * V' failed");

		print_matrix("tmp", tmp.get(), cd_.nSize, cd_.nSize, true);

		cuda_tools::UniquePtr<cuComplex> H_noise(cd_.nSize * fd_.frame_res());

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
			tmp.get(),
			cd_.nSize,
			&beta,
			H_noise.get(),
			fd_.frame_res());
		cuda_status = cudaDeviceSynchronize();
		assert(cuda_status == cudaSuccess);
		assert(cublas_status == CUBLAS_STATUS_SUCCESS && "H_noise = H * tmp failed");

		print_matrix("H_noise", H_noise.get(), fd_.frame_res(), cd_.nSize, true);

		subtract_frame_complex(H, H_noise.get(), H, fd_.frame_res() * cd_.nSize);

		average_complex_images(H, buffers_.gpu_input_buffer_.get(), fd_.frame_res(), cd_.nSize);
		cudaDeviceSynchronize();
		auto nppi_data = cuda_tools::NppiData(fd_.height, fd_.width);
		cuComplex constant = { static_cast<float>(cd_.nSize), 0 };
		cuda_tools::nppi_divide_by_constant(buffers_.gpu_input_buffer_.get(), nppi_data, constant);
		cudaDeviceSynchronize();

		/* cublas_status = cublasCgeam(cuda_tools::CublasHandle::instance(),
			CUBLAS_OP_T,
			CUBLAS_OP_N,
			fd_.height,
			fd_.width,
			&alpha,
			H + cd_.pindex * fd_.frame_size(),
			fd_.height,
			&beta,
			buffers_.gpu_input_buffer_.get(),
			fd_.height,
			buffers_.gpu_input_buffer_.get(),
			fd_.height);
		cuda_status = cudaDeviceSynchronize();
		assert(cuda_status == cudaSuccess);
		assert(cublas_status == CUBLAS_STATUS_SUCCESS && "Could not transpose final image"); */
		/* cublas_status = cublasCgeam(cuda_tools::CublasHandle::instance(),
			CUBLAS_OP_T,
			CUBLAS_OP_N,
			fd_.width,
			fd_.height,
			&alpha,
			buffers_.gpu_input_buffer_.get(),
			fd_.width,
			&beta,
			H,
			fd_.width,
			buffers_.gpu_input_buffer_.get(),
			fd_.height);
		cuda_status = cudaDeviceSynchronize();
		assert(cuda_status == cudaSuccess);
		std::cout << "status: " << (unsigned)cublas_status << std::endl;
		assert(cublas_status == CUBLAS_STATUS_SUCCESS && "Could not transpose final image"); */

		// print_matrix("OUTPUT", buffers_.gpu_input_buffer_.get(), fd_.height, fd_.width, false);

		 /*cudaMemcpy(buffers_.gpu_input_buffer_.get(),
			H + cd_.pindex * fd_.frame_size(),
			fd_.frame_size() * sizeof(cuComplex),
			cudaMemcpyDeviceToDevice); */
	});
}