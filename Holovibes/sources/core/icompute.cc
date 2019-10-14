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
#include "compute_bundles.hh"
#include "custom_exception.hh"
#include "unique_ptr.hh"
#include "pipe.hh"
#include "logger.hh"

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
		requested_output_size_(global::global_config.output_queue_max_size),
		unwrap_1d_requested_(false),
		unwrap_2d_requested_(false),
		autofocus_requested_(false),
		autocontrast_requested_(false),
		autocontrast_slice_xz_requested_(false),
		autocontrast_slice_yz_requested_(false),
		refresh_requested_(false),
		update_n_requested_(false),
		stft_update_roi_requested_(false),
		average_requested_(false),
		average_record_requested_(false),
		abort_construct_requested_(false),
		resize_requested_(false),
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
		new_fd.depth = 4;
		if (compute_desc_.img_acc_slice_yz_enabled)
		{
			auto fd_yz = new_fd;
			fd_yz.width = compute_desc_.nSize;
			gpu_img_acc_yz_.reset(new Queue(fd_yz, compute_desc_.img_acc_slice_yz_level, "AccumulationQueueYZ"));
			if (!gpu_img_acc_yz_)
				LOG_ERROR("Can't allocate queue");
		}
		if (compute_desc_.img_acc_slice_xz_enabled)
		{
			auto fd_xz = new_fd;
			fd_xz.height = compute_desc_.nSize;
			gpu_img_acc_xz_.reset(new Queue(fd_xz, compute_desc_.img_acc_slice_xz_level, "AccumulationQueueXZ"));
			if (!gpu_img_acc_xz_)
				LOG_ERROR("Can't allocate queue");
		}

		int inembed[1];
		int zone_size = input_.get_pixels();
			
		inembed[0] = compute_desc_.nSize;
		if (compute_desc_.croped_stft)
			zone_size = compute_desc_.getZoomedZone().area();

		stft_env_.plan1d_stft_.planMany(1, inembed,
			inembed, zone_size, 1,
			inembed, zone_size, 1,
			CUFFT_C2C, zone_size);

		camera::FrameDescriptor new_fd2 = input_.get_frame_desc();
		new_fd2.depth = 8;
		stft_env_.gpu_stft_queue_.reset(new Queue(new_fd2, compute_desc_.stft_level, "STFTQueue"));

		if (compute_desc_.ref_diff_enabled || compute_desc_.ref_sliding_enabled)
		{
			camera::FrameDescriptor new_fd3 = input_.get_frame_desc();
			new_fd3.depth = 8;
			/* Useless line. Maybe forgot gpu_ref_queue_ ?
			new Queue(new_fd3, compute_desc_.stft_level, "TakeRefQueue");
			*/
		}
		if (!buffers_.gpu_input_buffer_.resize(input_.get_pixels()))
			err++;
		int output_buffer_size = input_.get_pixels();
		if (compute_desc_.img_type == Composite)
			output_buffer_size *= 3;
		if (!buffers_.gpu_output_buffer_.resize(output_buffer_size))
			err++;
		buffers_.gpu_float_buffer_size_ = input_.get_pixels();
		if (compute_desc_.img_type == ImgType::Composite)
			buffers_.gpu_float_buffer_size_ *= 3;
		if (!buffers_.gpu_float_buffer_.resize(buffers_.gpu_float_buffer_size_))
			err++;
		
		if (err != 0)
			throw std::exception(cudaGetErrorString(cudaGetLastError()));

		// Setting the cufft plans to work on the default stream.
		cufftSetStream(plan2d_, static_cast<cudaStream_t>(0));
	}

	ICompute::~ICompute()
	{
		InfoManager::get_manager()->remove_info("Rendering Fps");
		InfoManager::get_manager()->remove_info("STFT Zone");
	}

	bool	ICompute::update_n_parameter(unsigned short n)
	{
		unsigned int err_count = 0;
		abort_construct_requested_ = false;
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
			/* This will resize cuts buffers: Some modifications are to be applied on opengl to work.

			if (compute_desc_.stft_view_enabled)
				request_stft_cuts_ = true; */
			stft_env_.gpu_stft_queue_.reset(new Queue(new_fd, n, "STFTQueue"));
			if (auto pipe = dynamic_cast<Pipe *>(this))
				pipe->get_fourier_transforms()->allocate_filter2d(n);
		}
		catch (std::exception&)
		{
			stft_env_.gpu_stft_queue_.reset();
			request_stft_cuts_ = false;
			request_delete_stft_cuts_ = true;
			request_queues();
			err_count++;
		}

		if (err_count != 0)
		{
			abort_construct_requested_ = true;
			allocation_failed(err_count, CustomException("error in update_n_parameters(n)", error_kind::fail_update));
			return false;
		}

		if (!buffers_.gpu_input_buffer_.resize(input_.get_pixels()))
			return false;
		notify_observers();
		return true;
	}

	void ICompute::request_queues()
	{
		if (request_stft_cuts_)
		{
			camera::FrameDescriptor fd_xz = output_.get_frame_desc();

			fd_xz.depth = sizeof(ushort);
			uint buffer_depth = sizeof(float);
			auto fd_yz = fd_xz;
			fd_xz.height = compute_desc_.nSize;
			fd_yz.width = compute_desc_.nSize;
			stft_env_.gpu_stft_slice_queue_xz.reset(new Queue(fd_xz, global::global_config.stft_cuts_output_buffer_size, "STFTCutXZ"));
			stft_env_.gpu_stft_slice_queue_yz.reset(new Queue(fd_yz, global::global_config.stft_cuts_output_buffer_size, "STFTCutYZ"));
			buffers_.gpu_float_cut_xz_.resize(fd_xz.frame_res() * buffer_depth);
			buffers_.gpu_float_cut_yz_.resize(fd_yz.frame_res() * buffer_depth);

			buffers_.gpu_ushort_cut_xz_.resize(fd_xz.frame_res());
			buffers_.gpu_ushort_cut_yz_.resize(fd_yz.frame_res());
			request_stft_cuts_ = false;
		}

		if (request_delete_stft_cuts_)
		{
			buffers_.gpu_float_cut_xz_.reset();
			buffers_.gpu_float_cut_yz_.reset();
			buffers_.gpu_ushort_cut_xz_.reset();
			buffers_.gpu_ushort_cut_yz_.reset();

			stft_env_.gpu_stft_slice_queue_xz.reset();
			stft_env_.gpu_stft_slice_queue_yz.reset();
			request_delete_stft_cuts_ = false;
		}
	}

	void	ICompute::delete_stft_slice_queue()
	{
		request_delete_stft_cuts_ = true;
		request_refresh();
	}

	void	ICompute::create_stft_slice_queue()
	{
		request_stft_cuts_ = true;
		request_refresh();
	}

	bool	ICompute::get_cuts_request()
	{
		return request_stft_cuts_;
	}

	bool	ICompute::get_cuts_delete_request()
	{
		return request_delete_stft_cuts_;
	}

	std::unique_ptr<Queue>&	ICompute::get_stft_slice_queue(int slice)
	{
		return slice ? stft_env_.gpu_stft_slice_queue_yz : stft_env_.gpu_stft_slice_queue_xz;
	}

	void ICompute::set_gpib_interface(std::shared_ptr<gpib::IVisaInterface> gpib_interface)
	{
		gpib_interface_ = gpib_interface;
	}

	void ICompute::allocation_failed(const int& err_count, std::exception& e)
	{
		LOG_ERROR(
			std::string("Pipe error:\n") +
			std::string("  message: ") + std::string(e.what()) + std::string("\n") +
			std::string("  err_count: ") + std::to_string(err_count) + std::string("\n\n")
		);
		notify_error_observers(e);
	}

	void ICompute::update_acc_parameter(
		std::unique_ptr<Queue>& queue,
		std::atomic<bool>& enabled,
		std::atomic<uint>& queue_length, 
		FrameDescriptor new_fd)
	{
		if (enabled && queue && queue->get_max_elts() == queue_length)
			return;
		queue = nullptr;
		if (enabled)
		{
			try
			{
				new_fd.depth = 4;
				queue.reset(new Queue(new_fd, queue_length, "Accumulation"));
				if (!queue)
					LOG_ERROR("couldn't allocate queue");
			}
			catch (std::exception&)
			{
				queue = nullptr;
				enabled = false;
				queue_length = 1;
				allocation_failed(1, CustomException("update_acc_parameter()", error_kind::fail_accumulation));
			}
		}
	}

	bool ICompute::get_request_refresh()
	{
		return refresh_requested_;
	}

	void ICompute::request_refresh()
	{
		refresh_requested_ = true;
	}

	void ICompute::request_acc_refresh()
	{
		update_acc_requested_ = true;
		request_refresh();
	}

	void ICompute::request_ref_diff_refresh()
	{
		update_ref_diff_requested_ = true;
		request_refresh();
	}

	void ICompute::request_termination()
	{
		termination_requested_ = true;
	}

	void ICompute::request_resize(unsigned int new_output_size, bool kill_raw_queue)
	{
		requested_output_size_ = new_output_size;
		resize_requested_ = true;
		kill_raw_queue_ = kill_raw_queue;
		request_refresh();
	}

	void ICompute::request_autocontrast(WindowKind kind)
	{
		if (kind == XYview)
			autocontrast_requested_ = true;
		else if (kind == XZview)
			autocontrast_slice_xz_requested_ = true;
		else
			autocontrast_slice_yz_requested_ = true;
	}

	void ICompute::request_filter2D_roi_update()
	{
		stft_update_roi_requested_ = true;
		request_update_n(compute_desc_.nSize);
	}

	void ICompute::request_filter2D_roi_end()
	{
		stft_update_roi_requested_ = false;
		request_update_n(compute_desc_.nSize);
		compute_desc_.log_scale_slice_xy_enabled = false;
		compute_desc_.shift_corners_enabled = true;
		notify_observers();

		if (auto pipe = dynamic_cast<Pipe*>(this))
			pipe->autocontrast_end_pipe(XYview);
	}

	void ICompute::request_autofocus()
	{
		autofocus_requested_ = true;
		autofocus_stop_requested_ = false;
		request_refresh();
	}

	void ICompute::request_autofocus_stop()
	{
		autofocus_stop_requested_ = true;
	}

	void ICompute::request_update_n(const unsigned short n)
	{
		update_n_requested_ = true;
		request_refresh();
	}

	void ICompute::request_update_unwrap_size(const unsigned size)
	{
		compute_desc_.unwrap_history_size = size;
		request_refresh();
	}

	void ICompute::request_unwrapping_1d(const bool value)
	{
		unwrap_1d_requested_ = value;
	}

	void ICompute::request_unwrapping_2d(const bool value)
	{
		unwrap_2d_requested_ = value;
	}

	void ICompute::request_average(
		ConcurrentDeque<Tuple4f>* output)
	{
		assert(output != nullptr);

		output->resize(compute_desc_.nSize);
		average_env_.average_output_ = output;

		average_requested_ = true;
		request_refresh();
	}

	void ICompute::request_average_stop()
	{
		average_requested_ = false;
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

		average_requested_ = true;
		average_record_requested_ = true;
		request_refresh();
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
				manager->insert_info(gui::InfoManager::InfoType::RENDERING_FPS, "OutputFps", std::to_string(fps) + " fps");
				long long voxelPerSecond = fps * output_fd.frame_res() * compute_desc_.nSize;
				manager->insert_info(gui::InfoManager::InfoType::OUTPUT_THROUGHPUT, "Output Throughput",
					std::to_string(static_cast<int>(voxelPerSecond / 1e6)) + " MVoxel/s");
				long long bytePerSecond = fps * input_.get_frame_desc().frame_size() * compute_desc_.stft_steps;
				manager->insert_info(gui::InfoManager::InfoType::INPUT_THROUGHPUT, "Input Throughput",
					std::to_string(static_cast<int>(bytePerSecond / 1e6)) + " MB/s");
			}
			past_time_ = time;
			frame_count_ = 0;
		}
	}
}
