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

#include "pipe.hh"
#include "config.hh"
#include "info_manager.hh"
#include "compute_descriptor.hh"
#include "queue.hh"
#include "compute_bundles.hh"
#include "compute_bundles_2d.hh"
#include "logger.hh"

#include "fft1.cuh"
#include "fft2.cuh"
#include "filter2D.cuh"
#include "stft.cuh"
#include "convolution.cuh"
#include "composite.cuh"
#include "tools.cuh"
#include "tools_conversion.cuh"
#include "tools_compute.cuh"
#include "tools.hh"
#include "contrast_correction.cuh"
#include "custom_exception.hh"

namespace holovibes
{
	using camera::FrameDescriptor;

	Pipe::Pipe(
		Queue& input,
		Queue& output,
		ComputeDescriptor& desc)
		: ICompute(input, output, desc)
		, fn_vect_()
	{
		image_accumulation_ = std::make_unique<compute::ImageAccumulation>(fn_vect_, image_acc_env_, buffers_, input.get_fd(), desc);
		fourier_transforms_ = std::make_unique<compute::FourierTransform>(fn_vect_, buffers_, input.get_fd(), desc, plan2d_, stft_env_);
		rendering_ = std::make_unique<compute::Rendering>(fn_vect_, buffers_, average_env_, image_acc_env_, stft_env_, desc, input.get_fd(), output.get_fd(), this);
		converts_ = std::make_unique<compute::Converts>(fn_vect_, buffers_, stft_env_, plan_unwrap_2d_, desc, input.get_fd(), output.get_fd());
		postprocess_ = std::make_unique<compute::Postprocessing>(fn_vect_, buffers_, input.get_fd(), desc);

		update_n_requested_ = true;

		try
		{
			refresh();
		}
		catch (const holovibes::CustomException& e)
		{
			// If refresh() fails the compute descriptor settings will be
			// changed to something that should make refresh() work
			// (ex: lowering the GPU memory usage)
			LOG_WARN("Pipe refresh failed, trying one more time with updated compute descriptor");
			try
			{
				refresh();
			}
			catch (const holovibes::CustomException& e)
			{
				// If it still didn't work holovibes is probably going to freeze
				// and the only thing you can do is restart it manually
				LOG_ERROR("Pipe could not be initialized");
				LOG_ERROR("You might want to restart holovibes");
				throw e;
			}
		}
	}

	Pipe::~Pipe()
	{
	}

	bool Pipe::make_requests()
	{
		bool success_allocation = true;

		if (output_resize_requested_)
		{
			output_.resize(requested_output_size_);
			output_resize_requested_ = false;
		}

		postprocess_->allocate_buffers();

		// Updating number of images
		if (update_n_requested_)
		{
			if (!update_n_parameter(cd_.nSize))
			{
				success_allocation = false;
				cd_.pindex = 0;
				cd_.nSize = 1;
				update_n_parameter(1);
				LOG_WARN("Updating #img failed, #img updated to 1");
			}
			update_n_requested_ = false;
		}

		const auto& input_fd = input_.get_fd();

		if (request_update_stft_steps_)
		{
			const uint batch_size = cd_.stft_steps * input_fd.frame_res();
			// We avoid the depth in the multiplication because the resize already take it into account
			buffers_.gpu_input_buffer_.resize(batch_size);

			long long int n[] = {input_fd.height, input_fd.width};

			plan2d_.XtplanMany(2,	// 2D
								n,	// Dimension of inner most & outer most dimension
								n,	// Storage dimension size
								1,	// Between two inputs (pixels) of same image distance is one
								input_fd.frame_res(), // Distance between 2 same index pixels of 2 images
								CUDA_C_32F, // Input type
								n, 1, input_fd.frame_res(), // Ouput layout same as input
								CUDA_C_32F, // Output type
								cd_.stft_steps, // Batch size
								CUDA_C_32F); // Computation type

			request_update_stft_steps_ = false;
		}

		if (request_disable_lens_view_)
		{
			fourier_transforms_->get_lens_queue().reset(nullptr);
			request_disable_lens_view_ = false;
		}

		// Allocating cuts queues
		make_cuts_requests();

		if (kill_raw_queue_requested_) // Destroy gpu raw queue
		{
			gpu_raw_queue_.reset(nullptr);
			kill_raw_queue_requested_ = false;
		}

		// Allocating accumulation queues/buffers
		image_accumulation_->allocate_accumulation_queues();

		return success_allocation;
	}

	void Pipe::refresh()
	{
		fn_vect_.clear();

		if (cd_.compute_mode == Computation::Direct)
		{
			update_n_requested_ = false;
			refresh_requested_ = false;
			insert_direct_enqueue_output();
			return;
		}

		// Aborting if allocation failed
		if (!make_requests())
		{
			refresh_requested_ = false;
			return;
		}

		const camera::FrameDescriptor& input_fd = input_.get_fd();

		/* Begin insertions */

		insert_wait_frames();

		insert_raw_view_enqueue();

		converts_->insert_complex_conversion(input_);

		// spatial transform
		fourier_transforms_->insert_fft();

		// time transform
		if (cd_.time_filter == TimeFilter::STFT)
		{
			fourier_transforms_->insert_stft();
		}
		else if (cd_.time_filter == TimeFilter::SVD)
		{
			fourier_transforms_->insert_eigenvalue_filter();
		}

		fourier_transforms_->insert_stft_cuts_view();

		// Used for phase increase
		fourier_transforms_->insert_store_p_frame();

		converts_->insert_to_float(unwrap_2d_requested_);

		postprocess_->insert_convolution();
		postprocess_->insert_renormalize();

		image_accumulation_->insert_image_accumulation();

		rendering_->insert_fft_shift();
		if (average_requested_)
			rendering_->insert_average(average_record_requested_);
		rendering_->insert_log();

		insert_request_autocontrast();
		rendering_->insert_contrast(autocontrast_requested_, autocontrast_slice_xz_requested_, autocontrast_slice_yz_requested_);

		fn_vect_.push_back([=]() { fps_count(); });

		converts_->insert_to_ushort();

		insert_hologram_enqueue_output();

		refresh_requested_ = false;
	}

	void Pipe::insert_raw_view_enqueue()
	{
		fn_vect_.push_back([&](){
			if (cd_.raw_view || cd_.record_raw)
			{
				input_.copy_multiple(*get_raw_queue(), cd_.stft_steps);
			}
		});
	}

	void Pipe::insert_wait_frames()
	{
		fn_vect_.push_back([&](){
			// Wait while the input queue is enough filled
			while (input_.get_size() < cd_.stft_steps);
		});
	}

	void Pipe::insert_direct_enqueue_output()
	{
		fn_vect_.push_back([&]()
		{
			if (!output_.enqueue(input_.get_start()))
				throw CustomException("Can't enqueue the input frame in output_queue", error_kind::fail_enqueue);
			input_.dequeue();
		});
	}

	void Pipe::insert_hologram_enqueue_output()
	{
		fn_vect_.push_back([&](){
			if (!output_.enqueue(buffers_.gpu_output_buffer_))
				throw CustomException("Can't enqueue the output frame in output_queue", error_kind::fail_enqueue);

			if (cd_.stft_view_enabled)
			{
				if (!stft_env_.gpu_stft_slice_queue_xz->enqueue(buffers_.gpu_ushort_cut_xz_.get()))
					throw CustomException("Can't enqueue the output xz frame in output xz queue", error_kind::fail_enqueue);
				if (!stft_env_.gpu_stft_slice_queue_yz->enqueue(buffers_.gpu_ushort_cut_yz_.get()))
					throw CustomException("Can't enqueue the output yz frame in output yz queue", error_kind::fail_enqueue);
			}
		});
	}

	void Pipe::insert_request_autocontrast()
	{
		if (cd_.contrast_enabled && cd_.contrast_auto_refresh)
			request_autocontrast(cd_.current_window);
	}

	void Pipe::exec()
	{
		if (global::global_config.flush_on_refresh)
			input_.clear();
		while (!termination_requested_)
		{
			try
			{
				if (input_.get_size() >= 1)
				{
					// Run the entire pipeline of calculation
					run_all();

					if (refresh_requested_)
						refresh();
				}
			}
			catch (CustomException& e)
			{
				pipe_error(1, e);
			}
		}
	}

	std::unique_ptr<Queue>& Pipe::get_lens_queue()
	{
		return fourier_transforms_->get_lens_queue();
	}

	compute::FourierTransform *Pipe::get_fourier_transforms()
	{
		return fourier_transforms_.get();
	}

	void Pipe::run_end_pipe(std::function<void()> function)
	{
		std::lock_guard<std::mutex> lock(functions_mutex_);
		functions_end_pipe_.push_back(function);
	}

	void Pipe::autocontrast_end_pipe(WindowKind kind)
	{
		run_end_pipe([this, kind]() {request_autocontrast(kind); });
	}

	void Pipe::run_all()
	{
		for (FnType& f : fn_vect_)
			f();
		{
			std::lock_guard<std::mutex> lock(functions_mutex_);
			for (FnType& f : functions_end_pipe_)
				f();
			functions_end_pipe_.clear();
		}
	}
}
