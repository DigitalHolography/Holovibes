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
#include "pipe.hh"

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
		gpu_lens_(nullptr),
		gpu_kernel_buffer_(nullptr),
		gpu_special_queue_(nullptr),
		gpu_stft_slice_queue_xz(nullptr),
		gpu_stft_slice_queue_yz(nullptr),
		gpu_ref_diff_queue_(nullptr),
		gpu_tmp_input_(nullptr),
		unwrap_1d_requested_(false),
		unwrap_2d_requested_(false),
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
		average_n_(0),
		past_time_(std::chrono::high_resolution_clock::now())
	{
		int err = 0;

		if (cudaMalloc(&gpu_lens_, input_.get_pixels() * sizeof(cufftComplex)) != cudaSuccess)
			err++;

		plan2d_.plan(
			input_.get_frame_desc().height,
			input_.get_frame_desc().width,
			CUFFT_C2C);

		if (compute_desc_.convolution_enabled
			|| compute_desc_.flowgraphy_enabled)
		{
			/* gpu_tmp_input */
			if (cudaMalloc<cufftComplex>(&gpu_tmp_input_,
				sizeof(cufftComplex)* input_.get_pixels() * compute_desc_.nsamples) != cudaSuccess)
				err++;
		}
		if (compute_desc_.convolution_enabled)
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
		if (compute_desc_.flowgraphy_enabled || compute_desc_.convolution_enabled)
		{
			/* gpu_tmp_input */
			if (cudaMalloc<cufftComplex>(&gpu_special_queue_,
				sizeof(cufftComplex)* input_.get_pixels() * compute_desc_.special_buffer_size) != cudaSuccess)
				err++;
		}

		camera::FrameDescriptor new_fd = input_.get_frame_desc();
		new_fd.depth = 4.f;
		if (compute_desc_.img_acc_slice_yz_enabled)
		{
			auto fd_yz = new_fd;
			fd_yz.width = compute_desc_.nsamples;
			gpu_img_acc_yz_.reset(new Queue(fd_yz, compute_desc_.img_acc_slice_yz_level.load(), "AccumulationQueueYZ"));
			if (!gpu_img_acc_yz_)
				std::cerr << "Error: can't allocate queue" << std::endl;
		}
		if (compute_desc_.img_acc_slice_xz_enabled)
		{
			auto fd_xz = new_fd;
			fd_xz.height = compute_desc_.nsamples;
			gpu_img_acc_xz_.reset(new Queue(fd_xz, compute_desc_.img_acc_slice_xz_level.load(), "AccumulationQueueXZ"));
			if (!gpu_img_acc_xz_)
				std::cerr << "Error: can't allocate queue" << std::endl;
		}

		int inembed[1] = { compute_desc_.nsamples };

		int zone_size = input_.get_pixels();
		if (compute_desc_.croped_stft)
			zone_size = compute_desc_.getZoomedZone().area();

		stft_env_.plan1d_stft_.planMany(1, inembed,
			inembed, zone_size, 1,
			inembed, zone_size, 1,
			CUFFT_C2C, zone_size);

		camera::FrameDescriptor new_fd2 = input_.get_frame_desc();
		new_fd2.depth = 8.f;
		stft_env_.gpu_stft_queue_.reset(new Queue(new_fd2, compute_desc_.stft_level.load(), "STFTQueue"));

		std::stringstream ss;
		ss << "(X1,Y1,X2,Y2) = (";
		if (compute_desc_.croped_stft)
		{
			auto zone = compute_desc_.getZoomedZone();
			stft_env_.gpu_cropped_stft_buf_.resize(zone.area() * compute_desc_.nsamples);
			ss << zone.x() << "," << zone.y() << "," << zone.right() << "," << zone.bottom() << ")";
		}
		else
			ss << "0,0," << new_fd.width - 1 << "," << new_fd.height - 1 << ")";

		InfoManager::get_manager()->insert_info(InfoManager::STFT_ZONE, "STFT Zone", ss.str());

		if (compute_desc_.ref_diff_enabled || compute_desc_.ref_sliding_enabled)
		{
			camera::FrameDescriptor new_fd3 = input_.get_frame_desc();
			new_fd3.depth = 8.f;
			/* Useless line. Maybe forgot gpu_ref_queue_ ?
			new Queue(new_fd3, compute_desc_.stft_level.load(), "TakeRefQueue");
			*/
		}
		int complex_pixels = sizeof(cufftComplex) * input_.get_pixels();

		if (cudaMalloc<cufftComplex>(&buffers_.gpu_input_buffer_, complex_pixels) != cudaSuccess)
			err++;
		if (cudaMalloc(&buffers_.gpu_output_buffer_, output_.get_frame_desc().depth * input_.get_pixels()) != cudaSuccess)
			err++;
		buffers_.gpu_float_buffer_size_ = sizeof(float) * input_.get_pixels();
		if (compute_desc_.img_type == ImgType::Composite)
			buffers_.gpu_float_buffer_size_ *= 3;
		if (cudaMalloc<float>(&buffers_.gpu_float_buffer_, buffers_.gpu_float_buffer_size_) != cudaSuccess)
			err++;
		if (err != 0)
			throw std::exception(cudaGetErrorString(cudaGetLastError()));
		
		if (err != 0)
			throw std::exception(cudaGetErrorString(cudaGetLastError()));
	}

	ICompute::~ICompute()
	{
		/* gpu_lens */
		cudaFree(gpu_lens_);

		/* gpu_special_queue */
		cudaFree(gpu_special_queue_);

		cudaFree(gpu_tmp_input_);

		/* gpu_kernel_buffer */
		cudaFree(gpu_kernel_buffer_);

		cudaFree(buffers_.gpu_float_cut_xz_);
		cudaFree(buffers_.gpu_float_cut_yz_);

		cudaFree(buffers_.gpu_ushort_cut_xz_);
		cudaFree(buffers_.gpu_ushort_cut_yz_);

		InfoManager::get_manager()->remove_info("Rendering Fps");
		InfoManager::get_manager()->remove_info("STFT Zone");
	}

	bool	ICompute::update_n_parameter(unsigned short n)
	{
		unsigned int err_count = 0;
		abort_construct_requested_.exchange(false);

		{
			std::lock_guard<std::mutex> Guard(stft_env_.stftGuard_);
			stft_env_.gpu_stft_buffer_.reset();
			stft_env_.plan1d_stft_.reset();
			/* CUFFT plan1d realloc */
			int inembed_stft[1] = { n };

			int zone_size = input_.get_pixels();
			if (compute_desc_.croped_stft)
				zone_size = compute_desc_.getZoomedZone().area();

			stft_env_.plan1d_stft_.planMany(1, inembed_stft,
				inembed_stft, zone_size, 1,
				inembed_stft, zone_size, 1,
				CUFFT_C2C, zone_size);
			stft_env_.gpu_stft_buffer_.resize(input_.get_pixels() * n);
		}

		stft_env_.gpu_stft_queue_.reset();


		camera::FrameDescriptor new_fd = input_.get_frame_desc();
		// gpu_stft_queue is a complex queue
		new_fd.depth = 8;
		try
		{
			if (compute_desc_.stft_view_enabled.load())
				update_stft_slice_queue();
			stft_env_.gpu_stft_queue_.reset(new Queue(new_fd, n, "STFTQueue"));
			if (compute_desc_.croped_stft)
				stft_env_.gpu_cropped_stft_buf_.resize(compute_desc_.getZoomedZone().area() * n);
			else
				stft_env_.gpu_cropped_stft_buf_.reset();
			std::cout << std::endl;
		}
		catch (std::exception&)
		{
			stft_env_.gpu_stft_queue_.reset();
			gpu_stft_slice_queue_xz.reset();
			gpu_stft_slice_queue_yz.reset();
			err_count++;
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
		if (!float_output_requested_ && !complex_output_requested_ && fqueue_)
		{
			delete fqueue_;
			fqueue_ = nullptr;
		}

		if (compute_desc_.convolution_enabled || compute_desc_.flowgraphy_enabled)
		{
			/* gpu_tmp_input */
			cudaFree(gpu_tmp_input_);
			if (cudaMalloc(&gpu_tmp_input_,	sizeof(cufftComplex) * input_.get_pixels() * compute_desc_.nsamples.load()) != CUDA_SUCCESS)
				err_count++;
		}
		if (compute_desc_.convolution_enabled)
		{
			/* kst_size */
			int size = static_cast<int>(compute_desc_.convo_matrix.size());
			/* gpu_kernel_buffer */
			cudaFree(gpu_kernel_buffer_);
			/* gpu_kernel_buffer */
			if (cudaMalloc(&gpu_kernel_buffer_, sizeof(float) * size) != CUDA_SUCCESS)
				err_count++;
			/* Build the kst 3x3 matrix */
			float *kst_complex_cpu = new float[size];
			for (int i = 0; i < size; ++i)
				kst_complex_cpu[i] = compute_desc_.convo_matrix[i];
			if (cudaMemcpy(gpu_kernel_buffer_, kst_complex_cpu, sizeof(float) * size,
				cudaMemcpyHostToDevice) != CUDA_SUCCESS)
				err_count++;
			delete[] kst_complex_cpu;
		}
		/* not deleted properly !!!!*/
		if (compute_desc_.flowgraphy_enabled || compute_desc_.convolution_enabled)
		{
			/* gpu_tmp_input */
			cudaFree(gpu_special_queue_);
			/* gpu_tmp_input */
			if (cudaMalloc(&gpu_special_queue_,	sizeof(cufftComplex)* input_.get_pixels() * compute_desc_.special_buffer_size) != CUDA_SUCCESS)
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
		std::unique_ptr<Queue>& queue,
		std::atomic<bool>& enabled,
		std::atomic<uint>& queue_length, 
		FrameDescriptor new_fd,
		float depth)
	{
		if (enabled && queue && queue->get_max_elts() == queue_length)
			return;
		queue = nullptr;
		if (enabled)
		{
			new_fd.depth = depth;
			try
			{
				queue.reset(new Queue(new_fd, queue_length, "Accumulation"));
				if (!queue)
					std::cout << "error: couldn't allocate queue" << std::endl;
			}
			catch (std::exception&)
			{
				queue = nullptr;
				enabled.exchange(false);
				queue_length.exchange(1);
				allocation_failed(1, CustomException("update_acc_parameter()", error_kind::fail_accumulation));
			}
		}
	}

	void ICompute::update_ref_diff_parameter()
	{
		if (gpu_ref_diff_queue_ != nullptr)
		{
			gpu_ref_diff_queue_.reset(nullptr);
			ref_diff_state_ = ref_state::ENQUEUE;
		}

		if (compute_desc_.ref_diff_enabled.load() || compute_desc_.ref_sliding_enabled.load())
		{
			camera::FrameDescriptor new_fd = input_.get_frame_desc();
			new_fd.depth = 8;
			try
			{
				gpu_ref_diff_queue_.reset(new Queue(new_fd, compute_desc_.ref_diff_level.load(), "TakeRefQueue"));
				gpu_ref_diff_queue_->set_display(false);
			}
			catch (std::exception&)
			{
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

	void ICompute::record_float(float *float_output, cudaStream_t stream)
	{
		// TODO: use stream in enqueue
		fqueue_->enqueue(float_output);
	}

	void ICompute::record_complex(cufftComplex *complex_output, cudaStream_t stream)
	{
		fqueue_->enqueue(complex_output);
	}

	void ICompute::handle_reference(cufftComplex *input, const unsigned int nframes)
	{
		if (ref_diff_state_ == ref_state::ENQUEUE)
		{
			queue_enqueue(input, gpu_ref_diff_queue_.get());
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
			queue_enqueue(input, gpu_ref_diff_queue_.get());
			ref_diff_counter--;
			if (ref_diff_counter == 0)
				ref_diff_state_ = ref_state::COMPUTE;
		}
		else if (ref_diff_state_ == ref_state::COMPUTE)
		{
			queue_enqueue(input, gpu_ref_diff_queue_.get());
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

		stft_env_.stft_frame_counter_--;
		bool b = false;
		if (stft_env_.stft_frame_counter_ == 0)
		{
			b = true;
			stft_env_.stft_frame_counter_ = compute_desc_.stft_steps;
		}
		std::lock_guard<std::mutex> Guard(stft_env_.stftGuard_);

		if (!compute_desc_.vibrometry_enabled)
		{
			stft(input,
				output,
				stft_env_.gpu_stft_buffer_,
				stft_env_.plan1d_stft_,
				compute_desc_.nsamples,
				compute_desc_.pindex,
				compute_desc_.pindex,
				compute_desc_.nsamples,
				input_.get_frame_desc().width,
				input_.get_frame_desc().height,
				b,
				compute_desc_.croped_stft,
				compute_desc_.getZoomedZone(),
				stft_env_.gpu_cropped_stft_buf_.get(),
				static_cast<cudaStream_t>(0));
		}
		else
		{
			/* q frame pointer */
			//cufftComplex* q = input + 1 * input_.get_frame_desc().frame_res();
			stft(
				input,
				static_cast<cufftComplex *>(stft_env_.gpu_stft_queue_->get_buffer()),
				stft_env_.gpu_stft_buffer_,
				stft_env_.plan1d_stft_,
				compute_desc_.nsamples.load(),
				compute_desc_.pindex.load(),
				compute_desc_.vibrometry_q.load(),
				compute_desc_.nsamples.load(),
				input_.get_frame_desc().width,
				input_.get_frame_desc().height,
				b,
				compute_desc_.croped_stft.load(),
				compute_desc_.getZoomedZone(),
				stft_env_.gpu_cropped_stft_buf_,
				static_cast<cudaStream_t>(0));
		}
		if (compute_desc_.stft_view_enabled.load() && b)
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
			stft_view_begin(stft_env_.gpu_stft_buffer_,
				buffers_.gpu_float_cut_xz_,
				buffers_.gpu_float_cut_yz_,
				mouse_posx,
				mouse_posy,
				mouse_posx + (compute_desc_.x_accu_enabled ? compute_desc_.x_acc_level.load() : 0),
				mouse_posy + (compute_desc_.y_accu_enabled ? compute_desc_.y_acc_level.load() : 0),
				width,
				height,
				compute_desc_.img_type.load(),
				compute_desc_.nsamples.load(),
				compute_desc_.img_acc_slice_xz_enabled.load() ? compute_desc_.img_acc_slice_xz_level.load() : 1,
				compute_desc_.img_acc_slice_yz_enabled.load() ? compute_desc_.img_acc_slice_yz_level.load() : 1,
				compute_desc_.img_type.load());
		}
		stft_env_.stft_handle_ = true;
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

	Queue* ICompute::get_lens_queue()
	{
		auto pipe = dynamic_cast<Pipe *>(this);
		if (pipe)
			return pipe->get_lens_queue();
		return nullptr;
	}
}
