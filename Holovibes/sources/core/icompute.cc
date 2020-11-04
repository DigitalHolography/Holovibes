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
#include "chart.cuh"
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
		ComputeDescriptor& cd)
		: cd_(cd),
		input_(input),
		output_(output),
		requested_output_size_(global::global_config.output_queue_max_size),
		past_time_(std::chrono::high_resolution_clock::now())
	{
		int err = 0;

		plan_unwrap_2d_.plan(
			input_.get_fd().height,
			input_.get_fd().width,
			CUFFT_C2C);

		const camera::FrameDescriptor& fd = input_.get_fd();
		long long int n[] = {fd.height, fd.width};

		plan2d_.XtplanMany(2,	// 2D
							n,	// Dimension of inner most & outer most dimension
							n,	// Storage dimension size
							1,	// Between two inputs (pixels) of same image distance is one
							fd.frame_res(), // Distance between 2 same index pixels of 2 images
							CUDA_C_32F, // Input type
							n, 1, fd.frame_res(), // Ouput layout same as input
							CUDA_C_32F, // Output type
							cd_.batch_size, // Batch size
							CUDA_C_32F); // Computation type

		int inembed[1];
		int zone_size = input_.get_frame_res();

		inembed[0] = cd_.time_filter_size;

		stft_env_.plan1d_stft.planMany(1, inembed,
			inembed, zone_size, 1,
			inembed, zone_size, 1,
			CUFFT_C2C, zone_size);

		camera::FrameDescriptor new_fd2 = input_.get_fd();
		new_fd2.depth = 8;
		stft_env_.gpu_time_filter_queue.reset(new Queue(new_fd2, cd_.time_filter_size, "STFTQueue"));

		if (!buffers_.gpu_spatial_filter_buffer.resize(cd_.batch_size * input_.get_fd().frame_res()))
			err++;

		int output_buffer_size = input_.get_frame_res();
		if (cd_.img_type == Composite)
			output_buffer_size *= 3;
		if (!buffers_.gpu_output_frame.resize(output_buffer_size))
			err++;
		buffers_.gpu_postprocess_frame_size = input_.get_frame_res();

		if (cd_.img_type == ImgType::Composite)
			buffers_.gpu_postprocess_frame_size *= 3;

		if (!buffers_.gpu_postprocess_frame.resize(buffers_.gpu_postprocess_frame_size))
			err++;

		// Init the gpu_p_frame with the size of input image
		if (!stft_env_.gpu_p_frame.resize(buffers_.gpu_postprocess_frame_size))
			err++;

		if (err != 0)
			throw std::exception(cudaGetErrorString(cudaGetLastError()));
	}

	ICompute::~ICompute()
	{
		InfoManager::get_manager()->remove_info("Rendering Fps");
	}

	bool ICompute::update_time_filter_size(const unsigned short time_filter_size)
	{
		unsigned int err_count = 0;
		{
			stft_env_.gpu_p_acc_buffer.reset();
			stft_env_.plan1d_stft.reset();
			/* CUFFT plan1d realloc */
			int inembed_stft[1] = { time_filter_size };

			int zone_size = input_.get_frame_res();

			stft_env_.plan1d_stft.planMany(1, inembed_stft,
				inembed_stft, zone_size, 1,
				inembed_stft, zone_size, 1,
				CUFFT_C2C, zone_size);
			stft_env_.gpu_p_acc_buffer.resize(input_.get_frame_res() * time_filter_size);

			// Pre allocate all the buffer only when n changes to avoid 1 allocation every frame
			stft_env_.pca_cov.reset();
			stft_env_.pca_tmp_buffer.reset();
			stft_env_.pca_eigen_values.reset();
			stft_env_.pca_dev_info.reset();

			stft_env_.pca_cov.resize(time_filter_size * time_filter_size);
			stft_env_.pca_tmp_buffer.resize(time_filter_size * time_filter_size);
			stft_env_.pca_eigen_values.resize(time_filter_size);
			stft_env_.pca_dev_info.resize(1);
		}

		stft_env_.gpu_time_filter_queue.reset();


		camera::FrameDescriptor new_fd = input_.get_fd();
		// gpu_stft_queue is a complex queue
		new_fd.depth = 8;
		try
		{
			/* This will resize cuts buffers: Some modifications are to be applied on opengl to work */
			stft_env_.gpu_time_filter_queue.reset(new Queue(new_fd, time_filter_size, "STFTQueue"));
		}
		catch (std::exception&)
		{
			stft_env_.gpu_time_filter_queue.reset();
			request_time_filter_cuts_ = false;
			request_delete_time_filter_cuts_ = true;
			make_cuts_requests();
			err_count++;
		}

		if (err_count != 0)
		{
			pipe_error(err_count, CustomException("error in update_time_filter_size(time_filter_size)", error_kind::fail_update));
			return false;
		}

		notify_observers();
		return true;
	}

	void ICompute::make_cuts_requests()
	{
		if (request_time_filter_cuts_)
		{
			camera::FrameDescriptor fd_xz = output_.get_fd();

			fd_xz.depth = sizeof(ushort);
			uint buffer_depth = sizeof(float);
			auto fd_yz = fd_xz;
			fd_xz.height = cd_.time_filter_size;
			fd_yz.width = cd_.time_filter_size;
			stft_env_.gpu_output_queue_xz.reset(new Queue(fd_xz, global::global_config.time_filter_cuts_output_buffer_size, "STFTCutXZ"));
			stft_env_.gpu_output_queue_yz.reset(new Queue(fd_yz, global::global_config.time_filter_cuts_output_buffer_size, "STFTCutYZ"));
			buffers_.gpu_postprocess_frame_xz.resize(fd_xz.frame_res());
			buffers_.gpu_postprocess_frame_yz.resize(fd_yz.frame_res());

			buffers_.gpu_output_frame_xz.resize(fd_xz.frame_res());
			buffers_.gpu_output_frame_yz.resize(fd_yz.frame_res());
			request_time_filter_cuts_ = false;
		}

		if (request_delete_time_filter_cuts_)
		{
			buffers_.gpu_postprocess_frame_xz.reset();
			buffers_.gpu_postprocess_frame_yz.reset();
			buffers_.gpu_output_frame_xz.reset();
			buffers_.gpu_output_frame_yz.reset();

			stft_env_.gpu_output_queue_xz.reset();
			stft_env_.gpu_output_queue_yz.reset();
			request_delete_time_filter_cuts_ = false;
		}
	}

	std::unique_ptr<Queue>& ICompute::get_raw_queue()
	{
		if (!gpu_raw_queue_ && (cd_.raw_view || cd_.record_raw))
		{
			auto fd = input_.get_fd();
			gpu_raw_queue_ = std::make_unique<Queue>(fd, output_.get_max_size(), "RawOutputQueue");
		}
		return gpu_raw_queue_;
	}

	void	ICompute::delete_stft_slice_queue()
	{
		request_delete_time_filter_cuts_ = true;
		request_refresh();
	}

	void ICompute::create_stft_slice_queue()
	{
		request_time_filter_cuts_ = true;
		request_refresh();
	}

	bool ICompute::get_cuts_request()
	{
		return request_time_filter_cuts_;
	}

	bool ICompute::get_cuts_delete_request()
	{
		return request_delete_time_filter_cuts_;
	}

	std::unique_ptr<Queue>&	ICompute::get_stft_slice_queue(int slice)
	{
		return slice ? stft_env_.gpu_output_queue_yz : stft_env_.gpu_output_queue_xz;
	}

	void ICompute::set_gpib_interface(std::shared_ptr<gpib::IVisaInterface> gpib_interface)
	{
		gpib_interface_ = gpib_interface;
	}

	void ICompute::pipe_error(const int& err_count, std::exception& e)
	{
		LOG_ERROR(
			std::string("Pipe error:\n") +
			std::string("  message: ") + std::string(e.what()) + std::string("\n") +
			std::string("  err_count: ") + std::to_string(err_count) + std::string("\n\n")
		);
		notify_error_observers(e);
	}

	bool ICompute::get_request_refresh()
	{
		return refresh_requested_;
	}

	void ICompute::request_refresh()
	{
		refresh_requested_ = true;
	}

	void ICompute::request_termination()
	{
		termination_requested_ = true;
	}

	void ICompute::request_resize(unsigned int new_output_size)
	{
		requested_output_size_ = new_output_size;
		output_resize_requested_ = true;
		request_refresh();
	}

	void ICompute::request_kill_raw_queue()
	{
		kill_raw_queue_requested_ = true;
		request_refresh();
	}

	void ICompute::request_autocontrast(WindowKind kind)
	{
		// Do not request anything if contrast is not enabled
		if (!cd_.contrast_enabled)
			return;

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
		request_update_time_filter_size();
	}

	void ICompute::request_filter2D_roi_end()
	{
		stft_update_roi_requested_ = false;
		request_update_time_filter_size();
		cd_.log_scale_slice_xy_enabled = false;
		notify_observers();

		if (auto pipe = dynamic_cast<Pipe*>(this))
			pipe->autocontrast_end_pipe(XYview);
	}

	void ICompute::request_update_time_filter_size()
	{
		update_time_filter_size_requested_ = true;
		request_refresh();
	}

	void ICompute::request_update_unwrap_size(const unsigned size)
	{
		cd_.unwrap_history_size = size;
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

	void ICompute::request_chart(
		ConcurrentDeque<Tuple4f>* output)
	{
		assert(output != nullptr);

		output->resize(cd_.time_filter_size);
		chart_env_.chart_output_ = output;

		chart_requested_ = true;
		request_refresh();
	}

	void ICompute::request_chart_stop()
	{
		chart_requested_ = false;
		request_refresh();
	}

	void ICompute::request_chart_record(
		ConcurrentDeque<Tuple4f>* output,
		const uint n)
	{
		assert(output != nullptr);
		assert(n != 0);

		chart_env_.chart_output_ = output;
		chart_env_.chart_n_ = n;

		chart_requested_ = true;
		chart_record_requested_ = true;
		request_refresh();
	}

	void ICompute::request_update_batch_size()
	{
		request_update_batch_size_ = true;
		request_refresh();
	}

	void ICompute::request_update_time_filter_stride()
	{
		request_update_time_filter_stride_ = true;
		request_refresh();
	}

	void ICompute::request_disable_lens_view()
	{
		request_disable_lens_view_ = true;
		request_refresh();
	}

	void ICompute::fps_count()
	{
		if (++frame_count_ >= 100)
		{
			auto time = std::chrono::high_resolution_clock::now();
			long long diff = std::chrono::duration_cast<std::chrono::milliseconds>(time - past_time_).count();
			InfoManager *manager = gui::InfoManager::get_manager();
			const camera::FrameDescriptor& output_fd = output_.get_fd();

			if (diff)
			{
				const long long fps = frame_count_ * 1000 / diff;
				manager->insert_info(gui::InfoManager::InfoType::RENDERING_FPS, "OutputFps", std::to_string(fps) + " fps");
				const long long voxelPerSecond = fps * output_fd.frame_res() * cd_.time_filter_size;
				manager->insert_info(gui::InfoManager::InfoType::OUTPUT_THROUGHPUT, "Output Throughput",
					std::to_string(static_cast<int>(voxelPerSecond / 1e6)) + " MVoxel/s");
				const long long bytePerSecond = fps * input_.get_fd().frame_size() * cd_.time_filter_stride;
				manager->insert_info(gui::InfoManager::InfoType::INPUT_THROUGHPUT, "Input Throughput",
					std::to_string(static_cast<int>(bytePerSecond / 1e6)) + " MB/s");
			}
			past_time_ = time;
			frame_count_ = 0;
		}
	}
}
