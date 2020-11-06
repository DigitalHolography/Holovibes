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
#include "pipeline_utils.hh"

namespace holovibes
{
	using camera::FrameDescriptor;

	Pipe::Pipe(
		Queue& input,
		Queue& output,
		ComputeDescriptor& desc)
		: ICompute(input, output, desc)
	{
		ConditionType batch_condition = [&]() -> bool { return batch_env_.batch_index == cd_.time_filter_stride; };

		fn_compute_vect_ = FunctionVector(batch_condition);
		fn_end_vect_ = FunctionVector(batch_condition);

		image_accumulation_ = std::make_unique<compute::ImageAccumulation>(fn_compute_vect_, image_acc_env_, buffers_, input.get_fd(), desc);
		fourier_transforms_ = std::make_unique<compute::FourierTransform>(fn_compute_vect_, buffers_, input.get_fd(), desc, plan2d_, batch_env_, time_filter_env_);
		rendering_ = std::make_unique<compute::Rendering>(fn_compute_vect_, buffers_, chart_env_, image_acc_env_, time_filter_env_, desc, input.get_fd(), output.get_fd(), this);
		converts_ = std::make_unique<compute::Converts>(fn_compute_vect_, buffers_, batch_env_, time_filter_env_, plan_unwrap_2d_, desc, input.get_fd(), output.get_fd());
		postprocess_ = std::make_unique<compute::Postprocessing>(fn_compute_vect_, buffers_, input.get_fd(), desc);

		update_time_filter_size_requested_ = true;

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
		// In order to have a better memory management, free all the ressources
		// that needs to be freed first and allocate the ressources that need
		// to be allocated in second

		bool success_allocation = true;

		/* Free buffers */
		if (cd_.convolution_changed && !cd_.convolution_enabled)
		{
			postprocess_->dispose();
			cd_.convolution_changed = false; // Aknowledge signal from gui
		}

		if (request_disable_lens_view_)
		{
			fourier_transforms_->get_lens_queue().reset(nullptr);
			request_disable_lens_view_ = false;
		}

		if (kill_raw_queue_requested_) // Destroy gpu raw queue
		{
			gpu_raw_queue_.reset(nullptr);
			kill_raw_queue_requested_ = false;
		}

		if (request_delete_time_filter_cuts_)
		{
			dispose_cuts();
			request_delete_time_filter_cuts_ = false;
		}

		image_accumulation_->dispose(); // done only if requested

		/* Allocate buffer */
		if (cd_.convolution_changed && cd_.convolution_enabled)
		{
			postprocess_->init();
			cd_.convolution_changed = false; // Aknowledge signal from gui
		}

		if (output_resize_requested_)
		{
			output_.resize(requested_output_size_);
			output_resize_requested_ = false;
		}

		// Updating number of images
		if (update_time_filter_size_requested_)
		{
			if (!update_time_filter_size(cd_.time_filter_size))
			{
				success_allocation = false;
				cd_.pindex = 0;
				cd_.time_filter_size = 1;
				update_time_filter_size(1);
				LOG_WARN("Updating #img failed, #img updated to 1");
			}
			update_time_filter_size_requested_ = false;
		}

		if (request_update_time_filter_stride_)
		{
			batch_env_.batch_index = 0;
			request_update_time_filter_stride_ = false;
		}

		if (request_update_batch_size_)
		{
			update_spatial_filter_parameters();
			request_update_batch_size_ = false;
		}

		if (request_time_filter_cuts_)
		{
			init_cuts();
			request_time_filter_cuts_ = false;
		}

		image_accumulation_->init(); // done only if requested

		return success_allocation;
	}

	void Pipe::refresh()
	{
		fn_compute_vect_.clear();

		if (cd_.compute_mode == Computation::Raw)
		{
			update_time_filter_size_requested_ = false;
			refresh_requested_ = false;
			insert_raw_enqueue_output();
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

		// Move frames from gpu_input_buffer to gpu_stft_queue (with respect to time_filter_stride)
		insert_transfer_for_time_filter();

		// time transform
		if (cd_.time_filter == TimeFilter::STFT)
		{
			fourier_transforms_->insert_stft();
		}
		else if (cd_.time_filter == TimeFilter::PCA)
		{
			fourier_transforms_->insert_eigenvalue_filter();
		}

		fourier_transforms_->insert_time_filter_cuts_view();

		// Used for phase increase
		fourier_transforms_->insert_store_p_frame();

		converts_->insert_to_float(unwrap_2d_requested_);

		postprocess_->insert_convolution();
		postprocess_->insert_renormalize();

		image_accumulation_->insert_image_accumulation();

		rendering_->insert_fft_shift();
		if (chart_requested_)
			rendering_->insert_chart(chart_record_requested_);
		rendering_->insert_log();

		insert_request_autocontrast();
		rendering_->insert_contrast(autocontrast_requested_, autocontrast_slice_xz_requested_, autocontrast_slice_yz_requested_);

		fn_compute_vect_.conditional_push_back([=]() { fps_count(); });

		converts_->insert_to_ushort();

		insert_hologram_enqueue_output();

		insert_reset_batch_index();

		refresh_requested_ = false;
	}

	void Pipe::insert_reset_batch_index()
	{
		fn_compute_vect_.push_back([&](){
			if (batch_env_.batch_index == cd_.time_filter_stride)
				batch_env_.batch_index = 0;
		});
	}

	void Pipe::copy_frames_for_recording(std::function<void()> copy_function)
	{
		// Start recording if requested
		if (cd_.request_recorder_copy_frames)
		{
			cd_.copy_frames_done = false;
			remaining_raw_frames_copy_ = cd_.nb_frames_record;
			cd_.request_recorder_copy_frames = false;
		}

		if (remaining_raw_frames_copy_ > 0)
		{
			copy_function();
			if (remaining_raw_frames_copy_ == 0)
				cd_.copy_frames_done = true;
		}
	}

	void Pipe::insert_raw_view_enqueue()
	{
		fn_compute_vect_.push_back([&](){
			// The raw view and raw recording can't be enabled at the same time
			if (cd_.is_recording && cd_.record_raw)
			{
				auto copy_lambda = [&]()
				{
					unsigned int raw_frames_copy = std::min(static_cast<unsigned int>(cd_.batch_size.load()), remaining_raw_frames_copy_);
					input_.copy_multiple(*get_raw_queue(), raw_frames_copy);
					remaining_raw_frames_copy_ -= raw_frames_copy;
				};
				copy_frames_for_recording(copy_lambda);
			}
			else if (cd_.raw_view)
				input_.copy_multiple(*get_raw_queue(), cd_.batch_size);
		});
	}

	void Pipe::insert_transfer_for_time_filter()
	{
		fn_compute_vect_.push_back([&]()
		{
			time_filter_env_.gpu_time_filter_queue->enqueue_multiple(buffers_.gpu_spatial_filter_buffer.get(), cd_.batch_size);
			batch_env_.batch_index += cd_.batch_size;
			assert(batch_env_.batch_index <= cd_.time_filter_stride);
		});
	}

	void Pipe::insert_wait_frames()
	{
		fn_compute_vect_.push_back([&](){
			// Wait while the input queue is enough filled
			while (input_.get_size() < cd_.batch_size);
		});
	}

	void Pipe::safe_enqueue_output(Queue& output_queue,
									  unsigned short* frame,
									  const std::string& error)
	{
		if (!output_queue.enqueue(frame))
			throw CustomException(error, error_kind::fail_enqueue);
	}

	void Pipe::enqueue_output(Queue& output_queue,
							  unsigned short* frame,
							  bool is_recording,
							  const std::string& error)
	{
		if (is_recording)
		{
			auto lambda = [&]() {
				safe_enqueue_output(output_queue, frame, error);
				remaining_raw_frames_copy_ -= 1;
			};
			copy_frames_for_recording(lambda);
		}
		else
			safe_enqueue_output(output_queue, frame, error);
	}

	void Pipe::insert_raw_enqueue_output()
	{
		fn_compute_vect_.push_back([&]()
		{
			enqueue_output(output_,
						   static_cast<unsigned short*>(input_.get_start()),
						   cd_.is_recording && !cd_.record_raw,
						   "Can't enqueue the input frame in output queue");
			input_.dequeue();
		});
	}

	void Pipe::insert_hologram_enqueue_output()
	{
		fn_compute_vect_.conditional_push_back([&](){
			enqueue_output(output_,
						   buffers_.gpu_output_frame.get(),
						   cd_.is_recording && !cd_.record_raw,
						   "Can't enqueue the output frame in output queue");

			// Always enqueue the cuts if enabled
			if (cd_.time_filter_cuts_enabled)
			{
				safe_enqueue_output(*time_filter_env_.gpu_output_queue_xz.get(),
					buffers_.gpu_output_frame_xz.get(),
					"Can't enqueue the output xz frame in output xz queue");

				safe_enqueue_output(*time_filter_env_.gpu_output_queue_yz.get(),
					buffers_.gpu_output_frame_yz.get(),
					"Can't enqueue the output yz frame in output yz queue");
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

	void Pipe::insert_fn_end_vect(std::function<void()> function)
	{
		std::lock_guard<std::mutex> lock(fn_end_vect_mutex_);
		fn_end_vect_.push_back(function);
	}

	void Pipe::autocontrast_end_pipe(WindowKind kind)
	{
		insert_fn_end_vect([this, kind]() {request_autocontrast(kind); });
	}

	void Pipe::run_all()
	{
		for (FnType& f : fn_compute_vect_)
			f();
		{
			std::lock_guard<std::mutex> lock(fn_end_vect_mutex_);
			for (FnType& f : fn_end_vect_)
				f();
			fn_end_vect_.clear();
		}
	}
}