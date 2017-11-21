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

#include <cufft.h>
#include <cassert>

#include "icompute.hh"
#include "fft1.cuh"
#include "fft2.cuh"
#include "stft.cuh"
#include "tools.cuh"
#include "contrast_correction.cuh"
#include "preprocessing.cuh"
#include "autofocus.cuh"
#include "average.cuh"
#include "interpolation.cuh"
#include "queue.hh"
#include "concurrent_deque.hh"
#include "compute_descriptor.hh"
#include "power_of_two.hh"
#include "info_manager.hh"
#include "tools_compute.cuh"
#include "vibrometry.cuh"
#include "compute_bundles.hh"
#include "custom_exception.hh"
#include "unique_ptr.hh"

namespace holovibes
{
	using gui::InfoManager;
	using camera::FrameDescriptor;


	ICompute::ICompute(
		Queue& input,
		Queue& output,
		ComputeDescriptor& desc)
		: compute_desc_(desc),
		input_(input),
		output_(output),
		unwrap_res_(nullptr),
		unwrap_res_2d_(nullptr),
		gpu_stft_buffer_(nullptr),
		gpu_lens_(nullptr),
		gpu_kernel_buffer_(nullptr),
		gpu_special_queue_(nullptr),
		gpu_stft_queue_(nullptr),
		gpu_stft_slice_queue_xz(nullptr),
		gpu_stft_slice_queue_yz(nullptr),
		gpu_ref_diff_queue_(nullptr),
		gpu_filter2d_buffer(nullptr),
		gpu_tmp_input_(nullptr),
		plan3d_(0),
		plan2d_(0),
		plan1d_(0),
		unwrap_1d_requested_(false),
		unwrap_2d_requested_(false),
		plan1d_stft_(0),
		autofocus_requested_(false),
		autocontrast_requested_(false),
		refresh_requested_(false),
		update_n_requested_(false),
		stft_update_roi_requested_(false),
		average_requested_(false),
		average_record_requested_(false),
		abort_construct_requested_(false),
		termination_requested_(false),
		update_acc_requested_(false),
		update_ref_diff_requested_(false),
		request_stft_cuts_(false),
		request_delete_stft_cuts_(false),
		average_output_(nullptr),
		ref_diff_state_(ref_state::ENQUEUE),
		ref_diff_counter(0),
		stft_frame_counter(1),
		average_n_(0),
		af_env_(),
		past_time_(std::chrono::high_resolution_clock::now()),
		gpu_float_cut_xz_(nullptr),
		gpu_float_cut_yz_(nullptr),
		stft_handle(false)
	{
		int err = 0;

		if (desc.compute_mode.load() == Computation::Direct)
			input_length_ = 1;
		else
			input_length_ = desc.nsamples.load();

		if (cudaMalloc(&gpu_lens_, input_.get_pixels() * sizeof(cufftComplex)) != cudaSuccess)
			err++;

		if (compute_desc_.algorithm.load() == Algorithm::FFT1
			|| compute_desc_.algorithm.load() == Algorithm::FFT2)
			cufftPlan3d(
				&plan3d_,
				input_length_,                  // NX
				input_.get_frame_desc().width,  // NY
				input_.get_frame_desc().height, // NZ
				CUFFT_C2C);

		cufftPlan2d(
			&plan2d_,
			input_.get_frame_desc().height,
			input_.get_frame_desc().width,
			CUFFT_C2C);

		/* CUFFT plan1d temporal*/
		int inembed[1] = { static_cast<int>(input_length_) };

		cufftPlanMany(&plan1d_, 1, inembed,
			inembed, input_.get_pixels(), 1,
			inembed, input_.get_pixels(), 1,
			CUFFT_C2C, input_.get_pixels());

		if (compute_desc_.convolution_enabled.load()
			|| compute_desc_.flowgraphy_enabled.load())
		{
			/* gpu_tmp_input */
			if (cudaMalloc<cufftComplex>(&gpu_tmp_input_,
				sizeof(cufftComplex)* input_.get_pixels() * compute_desc_.nsamples.load()) != cudaSuccess)
				err++;
		}
		if (compute_desc_.convolution_enabled.load())
		{
			/* kst_size */
			int size = static_cast<int>(compute_desc_.convo_matrix.size());
			/* Build the kst 3x3 matrix */
			float *kst_complex_cpu = new float[size];
			for (int i = 0; i < size; ++i)
			{
				kst_complex_cpu[i] = compute_desc_.convo_matrix[i];
				//kst_complex_cpu[i].y = 0;
			}
			/* gpu_kernel_buffer */
			if (cudaMalloc<float>(&gpu_kernel_buffer_, sizeof(float) * size) == cudaSuccess)
				cudaMemcpy(gpu_kernel_buffer_, kst_complex_cpu, sizeof(float) * size, cudaMemcpyHostToDevice);
			else
				err++;
			delete[] kst_complex_cpu;
		}
		if (compute_desc_.flowgraphy_enabled.load() || compute_desc_.convolution_enabled.load())
		{
			/* gpu_tmp_input */
			if (cudaMalloc<cufftComplex>(&gpu_special_queue_,
				sizeof(cufftComplex)* input_.get_pixels() * compute_desc_.special_buffer_size.load()) != cudaSuccess)
				err++;
		}

		camera::FrameDescriptor new_fd = input_.get_frame_desc();
		new_fd.depth = 4.f;
		if (compute_desc_.img_acc_slice_yz_enabled.load())
		{
			auto fd_yz = new_fd;
			fd_yz.width = compute_desc_.nsamples;
			gpu_img_acc_yz_ = new Queue(fd_yz, compute_desc_.img_acc_slice_yz_level.load(), "AccumulationQueueYZ");
			if (!gpu_img_acc_yz_)
				std::cerr << "Error: can't allocate queue" << std::endl;
		}
		if (compute_desc_.img_acc_slice_xz_enabled.load())
		{
			auto fd_xz = new_fd;
			fd_xz.height = compute_desc_.nsamples;
			gpu_img_acc_xz_ = new Queue(fd_xz, compute_desc_.img_acc_slice_xz_level.load(), "AccumulationQueueXZ");
			if (!gpu_img_acc_xz_)
				std::cerr << "Error: can't allocate queue" << std::endl;
		}

		if (compute_desc_.stft_enabled.load())
		{
			cufftPlanMany(&plan1d_stft_, 1, inembed,
				inembed, input_.get_pixels(), 1,
				inembed, input_.get_pixels(), 1,
				CUFFT_C2C, input_.get_pixels());

			camera::FrameDescriptor new_fd2 = input_.get_frame_desc();
			new_fd2.depth = 8.f;
			gpu_stft_queue_ = new Queue(new_fd2, compute_desc_.stft_level.load(), "STFTQueue");
		}

		if (compute_desc_.ref_diff_enabled.load() || compute_desc_.ref_sliding_enabled.load())
		{
			camera::FrameDescriptor new_fd3 = input_.get_frame_desc();
			new_fd3.depth = 8.f;
			/* Useless line. Maybe forgot gpu_ref_queue_ ?
			new Queue(new_fd3, compute_desc_.stft_level.load(), "TakeRefQueue");
			*/
		}

		if (compute_desc_.filter_2d_enabled.load())
		{
			if (cudaMalloc<cufftComplex>(&gpu_filter2d_buffer,
				sizeof(cufftComplex)* input_.get_pixels()) != cudaSuccess)
				err++;
		}
		if (err != 0)
			throw std::exception(cudaGetErrorString(cudaGetLastError()));
	}

	ICompute::~ICompute()
	{
		/* CUFFT plan1d */
		cufftDestroy(plan1d_);

		/* CUFFT plan2d */
		cufftDestroy(plan2d_);

		/* CUFFT plan3d */
		cufftDestroy(plan3d_);

		/* CUFFT plan1d for STFT */
		cufftDestroy(plan1d_stft_);

		/* gpu_lens */
		cudaFree(gpu_lens_);

		/* gpu_stft_buffer */
		cudaFree(gpu_stft_buffer_);

		/* gpu_special_queue */
		cudaFree(gpu_special_queue_);

		/* gpu_float_buffer_af_zone */
		cudaFree(af_env_.gpu_float_buffer_af_zone);

		/* gpu_input_buffer_tmp */
		cudaFree(af_env_.gpu_input_buffer_tmp);

		cudaFree(gpu_tmp_input_);

		/* gpu_kernel_buffer */
		cudaFree(gpu_kernel_buffer_);

		/* gpu_img_acc */
		delete gpu_img_acc_yz_;
		delete gpu_img_acc_xz_;

		/* gpu_stft_queue */
		gpu_stft_slice_queue_xz.reset(nullptr);
		gpu_stft_slice_queue_yz.reset(nullptr);

		delete gpu_stft_queue_;

		/* gpu_take_ref_queue */
		delete gpu_ref_diff_queue_;

		/* gpu_filter2d_buffer */
		cudaFree(gpu_filter2d_buffer);

		if (gpu_float_cut_xz_)	cudaFree(gpu_float_cut_xz_);
		if (gpu_float_cut_yz_)	cudaFree(gpu_float_cut_yz_);

		if (gpu_ushort_cut_xz_)	cudaFree(gpu_ushort_cut_xz_);
		if (gpu_ushort_cut_yz_)	cudaFree(gpu_ushort_cut_yz_);

		gui::InfoManager::get_manager()->remove_info("Rendering Fps");
	}

	bool	ICompute::update_n_parameter(unsigned short n)
	{
		unsigned int err_count = 0;
		abort_construct_requested_.exchange(false);

		/* if stft, we don't need to allocate more than one frame */
		if (!compute_desc_.stft_enabled.load())
			input_length_ = n;
		else
			input_length_ = 1;


		/* CUFFT plan3d realloc */
		cudaDestroy<cufftResult>(&plan3d_) ? ++err_count : 0;

		if (compute_desc_.algorithm.load() == Algorithm::FFT1
			|| compute_desc_.algorithm.load() == Algorithm::FFT2)
			cufftPlan3d(
				&plan3d_,
				input_length_,                  // NX
				input_.get_frame_desc().width,  // NY
				input_.get_frame_desc().height, // NZ
				CUFFT_C2C) ? ++err_count : 0;

		/* CUFFT plan1d realloc */
		cudaDestroy<cufftResult>(&plan1d_) ? ++err_count : 0;

		/* gpu_stft_buffer */
		//cudaDestroy<cudaError_t>(&gpu_stft_buffer_) ? ++err_count : 0;

		int inembed[1] = { static_cast<int>(input_length_) };

		cufftPlanMany(&plan1d_, 1, inembed,
			inembed, input_.get_pixels(), 1,
			inembed, input_.get_pixels(), 1,
			CUFFT_C2C, input_.get_pixels());

		{
			std::lock_guard<std::mutex> Guard(stftGuard);
			if (gpu_stft_buffer_ != nullptr)
			{
				cudaFree(gpu_stft_buffer_);
				gpu_stft_buffer_ = nullptr;
			}
			cudaDestroy<cufftResult>(&plan1d_stft_) ? ++err_count : 0;
			if (compute_desc_.stft_enabled.load())
			{
				/* CUFFT plan1d realloc */
				int inembed_stft[1] = { n };

				cufftPlanMany(&plan1d_stft_, 1, inembed_stft,
					inembed_stft, input_.get_pixels(), 1,
					inembed_stft, input_.get_pixels(), 1,
					CUFFT_C2C, input_.get_pixels());
				if (cudaMalloc(&gpu_stft_buffer_, sizeof(cufftComplex) * input_.get_pixels() * n) != CUDA_SUCCESS)
					err_count++;
			}
		}

		if (gpu_stft_queue_ != nullptr)
		{
			delete gpu_stft_queue_;
			gpu_stft_queue_ = nullptr;
		}

		if (compute_desc_.stft_enabled.load())
		{
			camera::FrameDescriptor new_fd = input_.get_frame_desc();
			new_fd.depth = 8;
			try
			{
				if (compute_desc_.stft_view_enabled.load())
					update_stft_slice_queue();
				gpu_stft_queue_ = new Queue(new_fd, n, "STFTQueue");

			}
			catch (std::exception&)
			{
				gpu_stft_queue_ = nullptr;
				gpu_stft_slice_queue_xz = nullptr;
				gpu_stft_slice_queue_yz = nullptr;
				err_count++;
			}
		}

		if (err_count != 0)
		{
			abort_construct_requested_.exchange(true);
			allocation_failed(err_count,
				static_cast<std::exception>(CustomException("error in update_n_parameters(n)", error_kind::fail_update)));
			return false;
		}
		return true;
	}

	void	ICompute::update_stft_slice_queue()
	{
		//std::lock_guard<std::mutex> Guard(gpu_stft_slice_queue_xz->getGuard());
		delete_stft_slice_queue();
		create_stft_slice_queue();
	}

	void	ICompute::delete_stft_slice_queue()
	{
		request_delete_stft_cuts_.exchange(true);
		request_refresh();
	}

	void	ICompute::create_stft_slice_queue()
	{
		request_stft_cuts_.exchange(true);
		request_refresh();
	}

	void	ICompute::create_3d_vision_queue()
	{
		request_3d_vision_.exchange(true);
		request_refresh();
	}

	void	ICompute::delete_3d_vision_queue()
	{
		request_delete_3d_vision_.exchange(true);
		request_refresh();
	}

	bool	ICompute::get_cuts_request()
	{
		return request_stft_cuts_.load();
	}

	bool	ICompute::get_cuts_delete_request()
	{
		return request_delete_stft_cuts_.load();
	}

	Queue&	ICompute::get_stft_slice_queue(int slice)
	{
		return slice ? *gpu_stft_slice_queue_yz : *gpu_stft_slice_queue_xz;
	}

	Queue& ICompute::get_3d_vision_queue()
	{
		return *gpu_3d_vision;
	}

	void ICompute::set_gpib_interface(std::shared_ptr<gpib::IVisaInterface> gpib_interface)
	{
		gpib_interface_ = gpib_interface;
	}

	void ICompute::refresh()
	{
		unsigned int err_count = 0;
		if (!float_output_requested_.load() && !complex_output_requested_.load() && fqueue_)
		{
			delete fqueue_;
			fqueue_ = nullptr;
		}

		if (compute_desc_.convolution_enabled.load()
			|| compute_desc_.flowgraphy_enabled.load())
		{
			/* gpu_tmp_input */
			cudaFree(gpu_tmp_input_);
			/* gpu_tmp_input */
			if (cudaMalloc<cufftComplex>(&gpu_tmp_input_,
				sizeof(cufftComplex) * input_.get_pixels() * compute_desc_.nsamples.load()) != CUDA_SUCCESS)
				err_count++;
		}
		if (compute_desc_.convolution_enabled.load())
		{
			/* kst_size */
			int size = static_cast<int>(compute_desc_.convo_matrix.size());
			/* gpu_kernel_buffer */
			cudaFree(gpu_kernel_buffer_);
			/* gpu_kernel_buffer */
			if (cudaMalloc<float>(&gpu_kernel_buffer_, sizeof(float) * size) != CUDA_SUCCESS)
				err_count++;
			/* Build the kst 3x3 matrix */
			float *kst_complex_cpu = new float[size];
			for (int i = 0; i < size; ++i)
			{
				kst_complex_cpu[i] = compute_desc_.convo_matrix[i];
				//kst_complex_cpu[i].y = 0;
			}
			if (cudaMemcpy(gpu_kernel_buffer_, kst_complex_cpu, sizeof(float) * size,
				cudaMemcpyHostToDevice) != CUDA_SUCCESS)
				err_count++;
			delete[] kst_complex_cpu;
		}
		/* not deleted properly !!!!*/
		if (compute_desc_.flowgraphy_enabled.load() || compute_desc_.convolution_enabled.load())
		{
			/* gpu_tmp_input */
			cudaFree(gpu_special_queue_);
			/* gpu_tmp_input */
			if (cudaMalloc<cufftComplex>(&gpu_special_queue_,
				sizeof(cufftComplex)* input_.get_pixels() *
				compute_desc_.special_buffer_size.load()) != CUDA_SUCCESS)
				err_count++;
		}

		if (gpu_filter2d_buffer != nullptr)
		{
			cudaFree(gpu_filter2d_buffer);
			gpu_filter2d_buffer = nullptr;
		}

		if (compute_desc_.filter_2d_enabled.load())
		{
			if (cudaMalloc<cufftComplex>(&gpu_filter2d_buffer, sizeof(cufftComplex) *
				input_.get_pixels()) != CUDA_SUCCESS)
				err_count++;
		}

		if (err_count != 0)
			allocation_failed(err_count, CustomException("error in refresh()", error_kind::fail_update));
	}

	void ICompute::allocation_failed(const int& err_count, std::exception& e)
	{
		const char *cuda_error = cudaGetErrorString(cudaGetLastError());
		std::cout
			<< "[ERROR] ICompute l" << __LINE__ << std::endl
			<< " error message: " << e.what()
			<< " err_count: " << err_count << std::endl
			<< " cudaError_t: " << cuda_error
			<< std::endl;
		notify_error_observers(e, cuda_error);
	}

	void ICompute::update_acc_parameter(
		Queue*& queue,
		std::atomic<bool>& enabled,
		std::atomic<uint>& queue_length, 
		FrameDescriptor new_fd,
		float depth)
	{
		if (enabled && queue && queue->get_max_elts() == queue_length)
			return;
		delete queue;
		queue = nullptr;
		if (enabled)
		{
			new_fd.depth = depth;
			try
			{
				queue = new Queue(new_fd, queue_length, "Accumulation");
				if (!queue)
					std::cout << "error: couldn't allocate queue" << std::endl;
			}
			catch (std::exception&)
			{
				queue = nullptr;
				enabled.exchange(false);
				enabled.exchange(1);
				allocation_failed(1, CustomException("update_acc_parameter()", error_kind::fail_accumulation));
			}
		}
	}

	void ICompute::update_ref_diff_parameter()
	{
		if (gpu_ref_diff_queue_ != nullptr)
		{
			delete  gpu_ref_diff_queue_;
			gpu_ref_diff_queue_ = nullptr;
			ref_diff_state_ = ref_state::ENQUEUE;

		}

		if (compute_desc_.ref_diff_enabled.load() || compute_desc_.ref_sliding_enabled.load())
		{
			camera::FrameDescriptor new_fd = input_.get_frame_desc();
			new_fd.depth = 8;
			try
			{
				gpu_ref_diff_queue_ = new Queue(new_fd, compute_desc_.ref_diff_level.load(), "TakeRefQueue");
				gpu_ref_diff_queue_->set_display(false);
			}
			catch (std::exception&)
			{
				gpu_ref_diff_queue_ = nullptr;
				allocation_failed(1, CustomException("update_acc_parameter()", error_kind::fail_reference));

			}
		}
	}

	bool ICompute::get_request_refresh()
	{
		return refresh_requested_.load();
	}

	void ICompute::request_refresh()
	{
		refresh_requested_.exchange(true);
	}

	void ICompute::request_acc_refresh()
	{
		update_acc_requested_.exchange(true);
		request_refresh();
	}

	void ICompute::request_ref_diff_refresh()
	{
		update_ref_diff_requested_.exchange(true);
		request_refresh();
	}

	void ICompute::request_float_output(Queue* fqueue)
	{
		fqueue_ = fqueue;
		float_output_requested_.exchange(true);
		request_refresh();
	}

	void ICompute::request_float_output_stop()
	{
		float_output_requested_.exchange(false);
		request_refresh();
	}

	void ICompute::request_complex_output(Queue* fqueue)
	{
		fqueue_ = fqueue;
		complex_output_requested_.exchange(true);
		request_refresh();
	}

	void ICompute::request_complex_output_stop()
	{
		complex_output_requested_.exchange(false);
		request_refresh();
	}

	void ICompute::request_termination()
	{
		termination_requested_.exchange(true);
	}

	void ICompute::request_autocontrast()
	{
		autocontrast_requested_.exchange(true);
	}

	void ICompute::request_filter2D_roi_update()
	{
		stft_update_roi_requested_.exchange(true);
		request_update_n(compute_desc_.nsamples.load());
	}

	void ICompute::request_filter2D_roi_end()
	{
		stft_update_roi_requested_.exchange(false);
		request_update_n(compute_desc_.nsamples.load());
		compute_desc_.log_scale_slice_xy_enabled.exchange(false);
		compute_desc_.shift_corners_enabled.exchange(true);
		notify_observers();
		autocontrast_requested_.exchange(true);
	}

	void ICompute::request_autofocus()
	{
		autofocus_requested_.exchange(true);
		autofocus_stop_requested_.exchange(false);
		request_refresh();
	}

	void ICompute::request_autofocus_stop()
	{
		autofocus_stop_requested_.exchange(true);
	}

	void ICompute::request_update_n(const unsigned short n)
	{
		update_n_requested_.exchange(true);
		request_refresh();
	}

	void ICompute::request_update_unwrap_size(const unsigned size)
	{
		compute_desc_.unwrap_history_size.exchange(size);
		request_refresh();
	}

	void ICompute::request_unwrapping_1d(const bool value)
	{
		unwrap_1d_requested_.exchange(value);
	}

	void ICompute::request_unwrapping_2d(const bool value)
	{
		unwrap_2d_requested_.exchange(value);
	}

	void ICompute::request_average(
		ConcurrentDeque<Tuple4f>* output)
	{
		assert(output != nullptr);

		if (compute_desc_.stft_enabled.load())
			output->resize(compute_desc_.nsamples.load());
		average_output_ = output;

		average_requested_.exchange(true);
		request_refresh();
	}

	void ICompute::request_average_stop()
	{
		average_requested_.exchange(false);
		request_refresh();
	}

	void ICompute::request_average_record(
		ConcurrentDeque<Tuple4f>* output,
		const uint n)
	{
		assert(output != nullptr);
		assert(n != 0);

		average_output_ = output;
		average_n_ = n;

		average_requested_.exchange(true);
		average_record_requested_.exchange(true);
		request_refresh();
	}

	void ICompute::autocontrast_caller(	float*				input,
										const uint			size,
										const uint			offset,
										ComputeDescriptor&	compute_desc,
										std::atomic<float>&	min,
										std::atomic<float>&	max,
										cudaStream_t		stream)
	{
		float contrast_min = 0.f;
		float contrast_max = 0.f;
		auto_contrast_correction(input, size, offset, &contrast_min, &contrast_max, stream);
		min.exchange(contrast_min);
		max.exchange(contrast_max);
		compute_desc.notify_observers();
	}

	void ICompute::record_float(float *float_output, cudaStream_t stream)
	{
		// TODO: use stream in enqueue
		fqueue_->enqueue(float_output, cudaMemcpyDeviceToDevice);
	}

	void ICompute::record_complex(cufftComplex *complex_output, cudaStream_t stream)
	{
		fqueue_->enqueue(complex_output, cudaMemcpyDeviceToDevice);
	}

	void ICompute::handle_reference(cufftComplex *input, const unsigned int nframes)
	{
		if (ref_diff_state_ == ref_state::ENQUEUE)
		{
			queue_enqueue(input, gpu_ref_diff_queue_);
			ref_diff_counter--;
			if (ref_diff_counter == 0)
			{
				ref_diff_state_ = ref_state::COMPUTE;
				if (compute_desc_.ref_diff_level.load() > 1)
					mean_images(static_cast<cufftComplex *>(gpu_ref_diff_queue_->get_buffer())
						, static_cast<cufftComplex *>(gpu_ref_diff_queue_->get_buffer()),
						compute_desc_.ref_diff_level.load(), input_.get_pixels());
			}
		}
		if (ref_diff_state_ == ref_state::COMPUTE)
		{
			substract_ref(input, static_cast<cufftComplex *>(gpu_ref_diff_queue_->get_buffer()),
				input_.get_frame_desc().frame_res(), nframes,
				static_cast<cudaStream_t>(0));
		}
	}

	void ICompute::handle_sliding_reference(cufftComplex *input, const unsigned int nframes)
	{
		if (ref_diff_state_ == ref_state::ENQUEUE)
		{
			queue_enqueue(input, gpu_ref_diff_queue_);
			ref_diff_counter--;
			if (ref_diff_counter == 0)
				ref_diff_state_ = ref_state::COMPUTE;
		}
		else if (ref_diff_state_ == ref_state::COMPUTE)
		{
			queue_enqueue(input, gpu_ref_diff_queue_);
			if (compute_desc_.ref_diff_level.load() > 1)
				mean_images(static_cast<cufftComplex *>(gpu_ref_diff_queue_->get_buffer())
					, static_cast<cufftComplex *>(gpu_ref_diff_queue_->get_buffer()),
					compute_desc_.ref_diff_level.load(), input_.get_pixels());
			substract_ref(input, static_cast<cufftComplex *>(gpu_ref_diff_queue_->get_buffer()),
				input_.get_frame_desc().frame_res(), nframes,
				static_cast<cudaStream_t>(0));
		}
	}

	void ICompute::stft_handler(cufftComplex* input, cufftComplex* output)
	{
		static ushort mouse_posx;
		static ushort mouse_posy;

		stft_frame_counter--;
		bool b = false;
		if (stft_frame_counter == 0)
		{
			b = true;
			stft_frame_counter = compute_desc_.stft_steps.load();
		}
		std::lock_guard<std::mutex> Guard(stftGuard);

		if (!compute_desc_.vibrometry_enabled.load())
		{
			stft(input,
				output,
				gpu_stft_buffer_,
				plan1d_stft_,
				compute_desc_.nsamples.load(),
				compute_desc_.pindex.load(),
				compute_desc_.pindex.load(),
				input_.get_frame_desc().frame_res(),
				b,
				static_cast<cudaStream_t>(0));
		}
		else
		{
			/* q frame pointer */
			//cufftComplex* q = input + 1 * input_.get_frame_desc().frame_res();
			stft(
				input,
				static_cast<cufftComplex *>(gpu_stft_queue_->get_buffer()),
				gpu_stft_buffer_,
				plan1d_stft_,
				compute_desc_.nsamples.load(),
				compute_desc_.pindex.load(),
				compute_desc_.vibrometry_q.load(),
				input_.get_frame_desc().frame_res(),
				b,
				static_cast<cudaStream_t>(0));
		}
		if (compute_desc_.stft_view_enabled.load() && b == true)
		{
			// Conservation of the coordinates when cursor is outside of the window
			units::PointFd cursorPos;
			compute_desc_.stftCursor(cursorPos, AccessMode::Get);
			const ushort width = input_.get_frame_desc().width;
			const ushort height = input_.get_frame_desc().height;
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
				compute_desc_.x_accu_enabled ? compute_desc_.x_accu_min_level.load() : mouse_posx,
				compute_desc_.y_accu_enabled ? compute_desc_.y_accu_min_level.load() : mouse_posy,
				compute_desc_.x_accu_enabled ? compute_desc_.x_accu_max_level.load() : mouse_posx,
				compute_desc_.y_accu_enabled ? compute_desc_.y_accu_max_level.load() : mouse_posy,
				width,
				height,
				compute_desc_.img_type.load(),
				compute_desc_.nsamples.load(),
				compute_desc_.img_acc_slice_xz_enabled.load() ? compute_desc_.img_acc_slice_xz_level.load() : 1,
				compute_desc_.img_acc_slice_yz_enabled.load() ? compute_desc_.img_acc_slice_yz_level.load() : 1,
				compute_desc_.img_type.load());
		}
		stft_handle = true;
	}

	void ICompute::queue_enqueue(void* input, Queue* queue)
	{
		queue->enqueue(input, cudaMemcpyDeviceToDevice);
	}

	void ICompute::average_caller(
		float* input,
		const unsigned int width,
		const unsigned int height,
		const units::RectFd& signal,
		const units::RectFd& noise,
		cudaStream_t stream)
	{
		average_output_->push_back(make_average_plot(input, width, height, signal, noise, stream));
	}

	void ICompute::average_record_caller(
		float* input,
		const unsigned int width,
		const unsigned int height,
		const units::RectFd& signal,
		const units::RectFd& noise,
		cudaStream_t stream)
	{
		if (average_n_ > 0)
		{
			average_output_->push_back(make_average_plot(input, width, height, signal, noise, stream));
			average_n_--;
		}
		else
		{
			average_n_ = 0;
			average_output_ = nullptr;
			request_refresh();
		}
	}

	void ICompute::average_stft_caller(
		cufftComplex* stft_buffer,
		const unsigned int width,
		const unsigned int height,
		const unsigned int width_roi,
		const unsigned int height_roi,
		units::RectFd& signal_zone,
		units::RectFd& noise_zone,
		const unsigned int nsamples,
		cudaStream_t stream)
	{
		cufftComplex*   cbuf;
		float*          fbuf;

		if (cudaMalloc<cufftComplex>(&cbuf, width * height * sizeof(cufftComplex)) != CUDA_SUCCESS)
		{
			std::cout << "[ERROR] Couldn't cudaMalloc average output" << std::endl;
			return;
		}
		if (cudaMalloc<float>(&fbuf, width * height * sizeof(float)) != CUDA_SUCCESS)
		{
			cudaFree(cbuf);
			std::cout << "[ERROR] Couldn't cudaMalloc average output" << std::endl;
			return;
		}

		for (unsigned i = 0; i < nsamples; ++i)
		{
			(*average_output_)[i] = (make_average_stft_plot(cbuf, fbuf, stft_buffer, width, height, width_roi, height_roi, signal_zone, noise_zone, i, nsamples, stream));
		}

		cudaFree(cbuf);
		cudaFree(fbuf);
	}

	void ICompute::fps_count()
	{
		if (++frame_count_ >= 100)
		{
			auto time = std::chrono::high_resolution_clock::now();
			long long diff = std::chrono::duration_cast<std::chrono::milliseconds>(time - past_time_).count();
			InfoManager *manager = gui::InfoManager::get_manager();
			const camera::FrameDescriptor& output_fd = output_.get_frame_desc();

			if (diff)
			{
				long long fps = frame_count_ * 1000 / diff;
				manager->insert_info(gui::InfoManager::InfoType::RENDERING_FPS, "OutputFps", std::to_string(fps) + std::string(" fps"));
				long long voxelPerSeconds = fps * output_fd.frame_res() * compute_desc_.nsamples.load();
				manager->insert_info(gui::InfoManager::InfoType::THROUGHPUT, "Throughput", std::to_string(static_cast<int>(voxelPerSeconds / 1e6)) + std::string(" MVoxel/s"));
			}
			past_time_ = time;
			frame_count_ = 0;
		}
	}

	void ICompute::autofocus_init()
	{
		// Autofocus needs to work on the same images. It will computes on copies.
		try
		{
			af_env_.state = af_state::RUNNING;
			if (compute_desc_.stft_enabled.load())
			{
				// Saving stft parameters
				af_env_.old_nsamples = compute_desc_.nsamples.load();
				af_env_.old_p = compute_desc_.pindex.load();
				af_env_.old_steps = compute_desc_.stft_steps.load();

				// Setting new parameters for faster autofocus
				af_env_.nsamples = 2;
				af_env_.p = 1;
				compute_desc_.nsamples.exchange(af_env_.nsamples);
				compute_desc_.pindex.exchange(af_env_.p);
				update_n_parameter(af_env_.nsamples);

				// Setting the steps and the frame_counter in order to call autofocus_caller only
				// once stft_queue_ is fully updated and stft is computed
				compute_desc_.stft_steps.exchange(compute_desc_.nsamples);
				stft_frame_counter = af_env_.nsamples;

				af_env_.stft_index = af_env_.nsamples - 1;
				af_env_.state = af_state::COPYING;

				notify_observers();
			}

			af_env_.gpu_frame_size = sizeof(cufftComplex) * input_.get_pixels();
			// When stft, we want to save 'nsamples' frame, in order to entirely fill the stft_queue_
			af_env_.gpu_input_size = af_env_.gpu_frame_size * compute_desc_.nsamples;
			cudaFree(af_env_.gpu_input_buffer_tmp);
			if (cudaMalloc(&af_env_.gpu_input_buffer_tmp, af_env_.gpu_input_size) != cudaSuccess)
				throw std::exception("Autofocus : cudaMalloc fail");

			// Wait input_length_ images in queue input_, before call make_contiguous_complex
			//	(case when autofocus called instantly after changing the number of image in non-stft mode)
			while (input_.get_current_elts() < input_length_)
				continue;

			const float ratio = compute_desc_.interp_lambda > 0 ? compute_desc_.lambda / compute_desc_.interp_lambda : 1;

			// If stft, it saves only one frames in the end of gpu_input_buffer_tmp
			make_contiguous_complex(
				input_,
				af_env_.gpu_input_buffer_tmp + af_env_.stft_index * input_.get_pixels(),
				input_length_);

			compute_desc_.autofocusZone(af_env_.zone, AccessMode::Get);
			/* Compute square af zone. */
			const unsigned int zone_width = af_env_.zone.width();
			const unsigned int zone_height = af_env_.zone.height();

			af_env_.af_square_size = upper_window_size(zone_width, zone_height);

			const unsigned int af_size = af_env_.af_square_size * af_env_.af_square_size;

			cudaFree(af_env_.gpu_float_buffer_af_zone);
			if (cudaMalloc(&af_env_.gpu_float_buffer_af_zone, af_size * sizeof(float)) != cudaSuccess)
				throw std::exception("Autofocus : cudaMalloc fail");
			/* Initialize z_*  */
			af_env_.z_min = compute_desc_.autofocus_z_min.load();
			af_env_.z_max = compute_desc_.autofocus_z_max.load();

			const float z_div = static_cast<float>(compute_desc_.autofocus_z_div.load());

			af_env_.z_step = (af_env_.z_max - af_env_.z_min) / z_div;

			af_env_.af_z = 0.0f;

			af_env_.z_iter = compute_desc_.autofocus_z_iter.load();
			af_env_.z = af_env_.z_min;
			af_env_.focus_metric_values.clear();
		}
		catch (std::exception e)
		{
			autofocus_reset();
			std::cout << e.what() << std::endl;
		}
	}

	void ICompute::autofocus_restore(cuComplex *input_buffer)
	{
		if (af_env_.state == af_state::RUNNING)
		{
			if (compute_desc_.stft_enabled.load())
			{
				af_env_.stft_index--;

				cudaMemcpy(input_buffer,
					af_env_.gpu_input_buffer_tmp + af_env_.stft_index * input_.get_pixels(),
					af_env_.gpu_frame_size,
					cudaMemcpyDeviceToDevice);

				// Resetting the stft_index just before the call of autofocus_caller
				if (af_env_.stft_index == 0)
					af_env_.stft_index = af_env_.nsamples;
			}
			else
				cudaMemcpy(input_buffer,
					af_env_.gpu_input_buffer_tmp,
					af_env_.gpu_input_size,
					cudaMemcpyDeviceToDevice);
		}
	}

	void ICompute::autofocus_caller(float* input, cudaStream_t stream)
	{
		// Since stft_frame_counter and stft_steps are resetted in the init, we cannot call autofocus_caller when the stft_queue_ is not fully updated
		if (compute_desc_.stft_enabled.load() && af_env_.stft_index != af_env_.nsamples)
		{
			autofocus_reset();
			std::cout << "Autofocus: shouldn't be called there. You should report this bug." << std::endl;
			return;
		}

		const camera::FrameDescriptor& input_fd = input_.get_frame_desc();

		// Copying the square zone into the tmp buffer
		frame_memcpy(input, af_env_.zone, input_fd.width, af_env_.gpu_float_buffer_af_zone, af_env_.af_square_size, stream);

		// Evaluating function
		const float focus_metric_value = focus_metric(af_env_.gpu_float_buffer_af_zone,
			af_env_.af_square_size,
			stream,
			compute_desc_.autofocus_size.load());

		if (!std::isnan(focus_metric_value))
			af_env_.focus_metric_values.push_back(focus_metric_value);

		af_env_.z += af_env_.z_step;

		// End of loop
		if (autofocus_stop_requested_.load() || af_env_.z > af_env_.z_max)
		{
			// Find max z
			auto biggest = std::max_element(af_env_.focus_metric_values.begin(), af_env_.focus_metric_values.end());
			const float z_div = static_cast<float>(compute_desc_.autofocus_z_div.load());

			/* Case the max has not been found. */
			if (biggest == af_env_.focus_metric_values.end())
			{
				// Restoring old stft parameters
				if (compute_desc_.stft_enabled.load())
				{
					compute_desc_.stft_steps.exchange(af_env_.old_steps);
					compute_desc_.nsamples.exchange(af_env_.old_nsamples);
					compute_desc_.pindex.exchange(af_env_.old_p);
					update_n_parameter(compute_desc_.nsamples.load());
				}
				autofocus_reset();
				std::cout << "Autofocus: Couldn't find a good value for z" << std::endl;
				request_refresh();
				return;
			}
			long long max_pos = std::distance(af_env_.focus_metric_values.begin(), biggest);

			// This is our temp max
			af_env_.af_z = af_env_.z_min + max_pos * af_env_.z_step;

			// Calculation of the new max/min, taking the old step
			af_env_.z_min = af_env_.af_z - af_env_.z_step;
			af_env_.z_max = af_env_.af_z + af_env_.z_step;

			// prepare next iter
			if (--af_env_.z_iter > 0)
			{
				af_env_.z = af_env_.z_min;
				af_env_.z_step = (af_env_.z_max - af_env_.z_min) / z_div;
				af_env_.focus_metric_values.clear();
			}
		}

		// End of autofocus, free resources and notify the new z
		if (autofocus_stop_requested_.load() || af_env_.z_iter <= 0)
		{
			// Restoring old stft parameters
			if (compute_desc_.stft_enabled.load())
			{
				compute_desc_.stft_steps.exchange(af_env_.old_steps);
				compute_desc_.nsamples.exchange(af_env_.old_nsamples);
				compute_desc_.pindex.exchange(af_env_.old_p);
				update_n_parameter(compute_desc_.nsamples.load());
			}

			compute_desc_.zdistance.exchange(af_env_.af_z);
			compute_desc_.notify_observers();

			autofocus_reset();
		}
		request_refresh();
	}

	void ICompute::autofocus_reset()
	{
		// if gpu_input_buffer_tmp is freed before is used by cudaMemcpyNoReturn
		cudaFree(af_env_.gpu_float_buffer_af_zone);
		cudaFree(af_env_.gpu_input_buffer_tmp);

		//Resetting af_env_ for next use
		af_env_.focus_metric_values.clear();
		af_env_.stft_index = 0;
		af_env_.state = af_state::STOPPED;
	}


	void ICompute::interpolation_caller(cuComplex *buffer,
		const int width,
		const int height,
		const float ratio,
		cudaStream_t stream)
	{
		tex_interpolation(buffer, width, height, ratio, stream);
	}

	Queue *ICompute::get_lens_queue()
	{
		if (!gpu_lens_queue_ && compute_desc_.gpu_lens_display_enabled)
		{
			auto fd = input_.get_frame_desc();
			fd.depth = 8;
			gpu_lens_queue_ = std::make_unique<Queue>(fd, 16, "GPU lens queue");
		}
		return gpu_lens_queue_.get();
	}
}
