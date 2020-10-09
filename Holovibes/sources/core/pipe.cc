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
#include "preprocessing.cuh"
#include "contrast_correction.cuh"

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
		rendering_ = std::make_unique<compute::Rendering>(fn_vect_, buffers_, average_env_, desc, input.get_fd(), output.get_fd(), this);
		converts_ = std::make_unique<compute::Converts>(fn_vect_, buffers_, stft_env_, plan2d_, desc, input.get_fd(), output.get_fd());
		preprocess_ = std::make_unique<compute::Preprocessing>(fn_vect_, buffers_, input.get_fd(), desc);
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
		if (resize_requested_)
		{
			output_.resize(requested_output_size_);
			resize_requested_ = false;
			if (kill_raw_queue_)
				gpu_raw_queue_.reset(nullptr);
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
		// Allocating cuts queues
		request_queues();

		// Allocating accumulation queues/buffers
		image_accumulation_->allocate_accumulation_queues();

		preprocess_->allocate_ref(update_ref_diff_requested_);


		return success_allocation;
	}

	void Pipe::refresh()
	{
		if (cd_.compute_mode == Computation::Direct)
		{
			fn_vect_.clear();
			update_n_requested_ = false;
			refresh_requested_ = false;
			return;
		}
		// else computation mode is hologram

		// Aborting if allocation failed
		if (!make_requests())
		{
			refresh_requested_ = false;
			return;
		}

		const camera::FrameDescriptor& input_fd = input_.get_fd();

		/* Clean current vector. */
		fn_vect_.clear();

		/* Build step by step the vector of function to compute */
		fn_vect_.push_back([=]() {make_contiguous_complex(input_, buffers_.gpu_input_buffer_); });

		preprocess_->insert_ref();

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

		// make computation between the p and the p + pAcc frames
		converts_->insert_to_float(unwrap_2d_requested_);

		postprocess_->insert_convolution();
		postprocess_->insert_renormalize();

		image_accumulation_->insert_image_accumulation();

		rendering_->insert_fft_shift();
		if (average_requested_)
			rendering_->insert_average(average_record_requested_);
		rendering_->insert_log();
		rendering_->insert_contrast(autocontrast_requested_, autocontrast_slice_xz_requested_, autocontrast_slice_yz_requested_);

		fn_vect_.push_back([=]() {fps_count(); });

		converts_->insert_to_ushort();

		refresh_requested_ = false;
	}

	void *Pipe::get_enqueue_buffer()
	{
		return buffers_.gpu_output_buffer_;
	}

	void Pipe::exec()
	{
		if (global::global_config.flush_on_refresh)
			input_.clear();
		while (!termination_requested_)
		{
			try
			{
				if (input_.get_current_elts() >= 1)
				{
					stft_env_.stft_handle_ = false;
					run_all();
					if (cd_.compute_mode == Hologram)
					{
						bool act = stft_env_.stft_frame_counter_ == cd_.stft_steps;
						if (act)
						{
							if (!output_.enqueue(get_enqueue_buffer()))
							{
								input_.dequeue();
								break;
							}
							if (cd_.stft_view_enabled)
							{
								stft_env_.gpu_stft_slice_queue_xz->enqueue(buffers_.gpu_ushort_cut_xz_.get());
								stft_env_.gpu_stft_slice_queue_yz->enqueue(buffers_.gpu_ushort_cut_yz_.get());
							}
						}
					}
					else if (!output_.enqueue(input_.get_start()))
					{
						input_.dequeue();
						break;
					}

					if (cd_.compute_mode == Hologram && cd_.raw_view || cd_.record_raw)
					{
						if (!get_raw_queue()->enqueue(input_.get_start()))
						{
							input_.dequeue();
							break;
						}
					}
					input_.dequeue();
					if (refresh_requested_)
						refresh();
				}
			}
			catch (CustomException& e)
			{
				allocation_failed(1, e);
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
		{
			f();
			if (stft_env_.stft_frame_counter_ != cd_.stft_steps && stft_env_.stft_handle_)
				break;
		}
		{
			std::lock_guard<std::mutex> lock(functions_mutex_);
			for (FnType& f : functions_end_pipe_)
				f();
			functions_end_pipe_.clear();
		}
	}

	std::unique_ptr<Queue>& Pipe::get_raw_queue()
	{
		if (!gpu_raw_queue_ && (cd_.raw_view || cd_.record_raw))
		{
			auto fd = input_.get_fd();
			gpu_raw_queue_ = std::make_unique<Queue>(fd, output_.get_max_elts(), "RawOutputQueue");
		}
		return gpu_raw_queue_;
	}
}
