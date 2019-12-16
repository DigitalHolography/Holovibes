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
#include "autofocus.cuh"
#include "tools_conversion.cuh"
#include "tools_compute.cuh"
#include "tools.hh"
#include "preprocessing.cuh"
#include "contrast_correction.cuh"
#include "interpolation.cuh"

namespace holovibes
{
	using camera::FrameDescriptor;

	Pipe::Pipe(
		Queue& input,
		Queue& output,
		ComputeDescriptor& desc)
		: ICompute(input, output, desc)
		, fn_vect_()
		, detect_intensity_(fn_vect_, buffers_, input.get_frame_desc(), desc)
	{
		stabilization_ = std::make_unique<compute::Stabilization>(fn_vect_, buffers_, input.get_frame_desc(), desc);
		autofocus_ = std::make_unique<compute::Autofocus>(fn_vect_, buffers_, input_, desc, this);
		fourier_transforms_ = std::make_unique<compute::FourierTransform>(fn_vect_, buffers_, autofocus_, input.get_frame_desc(), desc, plan2d_, stft_env_);
		rendering_ = std::make_unique<compute::Rendering>(fn_vect_, buffers_, average_env_, desc, input.get_frame_desc(), output.get_frame_desc(), this);
		converts_ = std::make_unique<compute::Converts>(fn_vect_, buffers_, stft_env_, plan2d_, desc, input.get_frame_desc(), output.get_frame_desc());
		preprocess_ = std::make_unique<compute::Preprocessing>(fn_vect_, buffers_, input.get_frame_desc(), desc);
		postprocess_ = std::make_unique<compute::Postprocessing>(fn_vect_, buffers_, input.get_frame_desc(), desc);
		aberration_ = std::make_unique<compute::Aberration>(buffers_, input.get_frame_desc(), desc, fourier_transforms_->get_lens_queue().get());

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


	// This functions appears to be useless
	// because the conversions short -> complex -> float -> short are unnecessary
	// and should be optimized a lot.
	// But if it's removed, direct data recording drops frames, and the integrity of the frame is lost
	/*void Pipe::direct_refresh()
	{
		const camera::FrameDescriptor& input_fd = input_.get_frame_desc();
		const camera::FrameDescriptor& output_fd = output_.get_frame_desc();

		if (abort_construct_requested_)
		{
			refresh_requested_.exchange(false);
			return;
		}
		fn_vect_.push_back([=]() {
			make_contiguous_complex(
				input_,
				buffers_.gpu_input_buffer_);
		});
		fn_vect_.push_back([=]() {
			complex_to_modulus(
				buffers_.gpu_input_buffer_,
				buffers_.gpu_float_buffer_,
				nullptr,
				0,
				0,
				input_fd.frame_res());
		});
		fn_vect_.push_back([=]() {
			float_to_ushort(
				buffers_.gpu_float_buffer_,
				buffers_.gpu_output_buffer_,
				input_fd.frame_res(),
				output_fd.depth);
		});
	} */

	void Pipe::refresh()
	{
		if (compute_desc_.compute_mode == Computation::Direct)
		{
			fn_vect_.clear();
			// Useless function, that has no incidence on output, but maybe fixes a displaying bug of direct hologram
			//direct_refresh();
			update_n_requested_ = false;
			refresh_requested_ = false;
			return;
		}

		if (resize_requested_)
		{
			output_.resize(requested_output_size_);
			resize_requested_ = false;
			if (kill_raw_queue_)
			{
				gpu_raw_queue_.reset(nullptr);
			}
		}

		postprocess_->allocate_buffers();

		const camera::FrameDescriptor& input_fd = input_.get_frame_desc();

		/* Clean current vector. */
		fn_vect_.clear();

		// Updating number of images
		if (update_n_requested_)
		{
			if (!update_n_parameter(compute_desc_.nSize))
			{
				compute_desc_.pindex = 0;
				compute_desc_.nSize = 1;
				update_n_parameter(1);
				LOG_WARN("Updating #img failed, #img updated to 1");
			}
			update_n_requested_ = false;
		}
		// Allocating cuts queues
		request_queues();

		// Allocating accumulation queues/buffers
		if (update_acc_requested_)
		{
			FrameDescriptor new_fd_xz = input_.get_frame_desc();
			FrameDescriptor new_fd_yz = input_.get_frame_desc();
			new_fd_xz.height = compute_desc_.nSize;
			new_fd_yz.width = compute_desc_.nSize;
			update_acc_parameter(gpu_img_acc_yz_, compute_desc_.img_acc_slice_yz_enabled, compute_desc_.img_acc_slice_yz_level, new_fd_yz);
			update_acc_parameter(gpu_img_acc_xz_, compute_desc_.img_acc_slice_xz_enabled, compute_desc_.img_acc_slice_xz_level, new_fd_xz);
			update_acc_requested_ = false;
		}

		preprocess_->allocate_ref(update_ref_diff_requested_);

		// Aborting if allocation failed
		if (abort_construct_requested_)
		{
			refresh_requested_ = false;
			return;
		}

		if (autofocus_requested_ && autofocus_->get_state() == compute::af_state::STOPPED)
		{
			autofocus_->insert_init();
			autofocus_requested_ = false;
			return;
		}

		// Converting input_ to complex.
		if (autofocus_->get_state() == compute::af_state::STOPPED)
		{
			fn_vect_.push_back([=]() {make_contiguous_complex(input_, buffers_.gpu_input_buffer_); });
		}
		else if (autofocus_->get_state() == compute::af_state::COPYING)
		{
			autofocus_->insert_copy();
			return;
		}

		detect_intensity_.insert_post_contiguous_complex();

		autofocus_->insert_restore();

		preprocess_->insert_frame_normalization();
		preprocess_->insert_interpolation();
		preprocess_->insert_ref();

		preprocess_->insert_pre_fft_shift();

		fourier_transforms_->insert_fft();
		fourier_transforms_->insert_stft();

		aberration_->enqueue(fn_vect_);

		converts_->insert_to_float(unwrap_2d_requested_);

		postprocess_->insert_convolution();
		postprocess_->insert_renormalize();
		//TODO : apply convolution to XZ YZ cuts

		stabilization_->insert_post_img_type();

		// Inserts the output buffers into the accumulation queues
		// and rewrite the average into the output buffer
		// For the XY view, this happens in stabilization.cc,
		// as the average is used in the intermediate computations
		if (compute_desc_.img_acc_slice_yz_enabled)
			enqueue_buffer(gpu_img_acc_yz_.get(),
				static_cast<float *>(buffers_.gpu_float_cut_yz_.get()),
				compute_desc_.img_acc_slice_yz_level,
				input_fd.height * compute_desc_.nSize);
		if (compute_desc_.img_acc_slice_xz_enabled)
			enqueue_buffer(gpu_img_acc_xz_.get(),
				static_cast<float *>(buffers_.gpu_float_cut_xz_.get()),
				compute_desc_.img_acc_slice_xz_level,
				input_fd.width * compute_desc_.nSize);

		rendering_->insert_post_fft_shift();
		if (average_requested_)
			rendering_->insert_average(average_record_requested_);
		rendering_->insert_log();
		rendering_->insert_contrast(autocontrast_requested_, autocontrast_slice_xz_requested_, autocontrast_slice_yz_requested_);
		autofocus_->insert_autofocus();

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
					if (compute_desc_.compute_mode == Hologram)
					{
						bool act = stft_env_.stft_frame_counter_ == compute_desc_.stft_steps;
						if (act)
						{
							if (!output_.enqueue(get_enqueue_buffer()))
							{
								input_.dequeue();
								break;
							}
							if (compute_desc_.stft_view_enabled)
							{
								queue_enqueue(buffers_.gpu_ushort_cut_xz_.get(), stft_env_.gpu_stft_slice_queue_xz.get());
								queue_enqueue(buffers_.gpu_ushort_cut_yz_.get(), stft_env_.gpu_stft_slice_queue_yz.get());
							}
						}
					}
					else if (!output_.enqueue(input_.get_start()))
					{
						input_.dequeue();
						break;
					}

					if (compute_desc_.compute_mode == Hologram && compute_desc_.raw_view || compute_desc_.record_raw)
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

	void Pipe::enqueue_buffer(Queue* queue, float *buffer, uint nb_images, uint nb_pixels)
	{
		if (!queue)
		{
			LOG_ERROR("Error: queue is null");
			return;
		}
		/*Add image to phase accumulation buffer*/

		fn_vect_.push_back([=]() {queue_enqueue(buffer, queue); });
		fn_vect_.push_back([=]() {
			accumulate_images(
				static_cast<float *>(queue->get_buffer()),
				buffer,
				queue->get_start_index(),
				queue->get_max_elts(),
				nb_images,
				nb_pixels);
		});
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
		request_autocontrast(kind);
		run_end_pipe([this, kind]() {request_autocontrast(kind); });
	}

	void Pipe::run_all()
	{
		for (FnType& f : fn_vect_)
		{
			f();
			if (stft_env_.stft_frame_counter_ != compute_desc_.stft_steps && stft_env_.stft_handle_)
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
		if (!gpu_raw_queue_ && (compute_desc_.raw_view || compute_desc_.record_raw))
		{
			auto fd = input_.get_frame_desc();
			gpu_raw_queue_ = std::make_unique<Queue>(fd, output_.get_max_elts(), "RawOutputQueue");
		}
		return gpu_raw_queue_;
	}
}
