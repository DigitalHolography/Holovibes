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
		past_time_(std::chrono::high_resolution_clock::now())
	{
		int err = 0;

		plan2d_.plan(
			input_.get_frame_desc().height,
			input_.get_frame_desc().width,
			CUFFT_C2C);

		camera::FrameDescriptor new_fd = input_.get_frame_desc();
		new_fd.depth = 4.f;
		if (compute_desc_.img_acc_slice_yz_enabled)
		{
			auto fd_yz = new_fd;
			fd_yz.width = compute_desc_.nsamples;
			gpu_img_acc_yz_.reset(new Queue(fd_yz, compute_desc_.img_acc_slice_yz_level, "AccumulationQueueYZ"));
			if (!gpu_img_acc_yz_)
				std::cerr << "Error: can't allocate queue" << std::endl;
		}
		if (compute_desc_.img_acc_slice_xz_enabled)
		{
			auto fd_xz = new_fd;
			fd_xz.height = compute_desc_.nsamples;
			gpu_img_acc_xz_.reset(new Queue(fd_xz, compute_desc_.img_acc_slice_xz_level, "AccumulationQueueXZ"));
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
		stft_env_.gpu_stft_queue_.reset(new Queue(new_fd2, compute_desc_.stft_level, "STFTQueue"));

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

		// Setting the cufft plans to work on the default stream.
		cufftSetStream(plan2d_, static_cast<cudaStream_t>(0));
	}

	ICompute::~ICompute()
	{
		cudaFree(buffers_.gpu_input_buffer_);

		cudaFree(buffers_.gpu_float_cut_xz_);
		cudaFree(buffers_.gpu_float_cut_yz_);
		cudaFree(buffers_.gpu_float_buffer_);

		cudaFree(buffers_.gpu_ushort_cut_xz_);
		cudaFree(buffers_.gpu_ushort_cut_yz_);
		cudaFree(buffers_.gpu_output_buffer_);

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
			stft_env_.gpu_stft_slice_queue_xz.reset();
			stft_env_.gpu_stft_slice_queue_yz.reset();
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
		return slice ? *stft_env_.gpu_stft_slice_queue_yz : *stft_env_.gpu_stft_slice_queue_xz;
	}

	void ICompute::set_gpib_interface(std::shared_ptr<gpib::IVisaInterface> gpib_interface)
	{
		gpib_interface_ = gpib_interface;
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
		fqueue_.reset(fqueue);
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
		fqueue_.reset(fqueue);
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
		average_env_.average_output_ = output;

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

		average_env_.average_output_ = output;
		average_env_.average_n_ = n;

		average_requested_.exchange(true);
		average_record_requested_.exchange(true);
		request_refresh();
	}

	void ICompute::record(void *output, cudaStream_t stream)
	{
		//TODo: use stream
		fqueue_->enqueue(output);
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
