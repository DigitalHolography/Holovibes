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

#include "fft1.cuh"
#include "fft2.cuh"
#include "filter2D.cuh"
#include "stft.cuh"
#include "convolution.cuh"
#include "composite.cuh"
#include "flowgraphy.cuh"
#include "tools.cuh"
#include "autofocus.cuh"
#include "tools_conversion.cuh"
#include "tools_compute.cuh"
#include "tools.hh"
#include "preprocessing.cuh"
#include "contrast_correction.cuh"
#include "vibrometry.cuh"
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
		, detect_intensity_(fn_vect_, buffers_.gpu_input_buffer_, input.get_frame_desc(), desc)
	{
		stabilization_ = std::make_unique<compute::Stabilization>(fn_vect_, buffers_.gpu_float_buffer_, input.get_frame_desc(), desc);
		autofocus_ = std::make_unique<compute::Autofocus>(fn_vect_, buffers_.gpu_float_buffer_, buffers_.gpu_input_buffer_, input_, desc, this);
		fourier_transforms_ = std::make_unique<compute::FourierTransform>(fn_vect_, buffers_, autofocus_,	input.get_frame_desc(), desc, plan2d_, stft_env_);
		contrast_ = std::make_unique<compute::Contrast>(fn_vect_, buffers_, desc, output.get_frame_desc(), gpu_3d_vision, autocontrast_requested_);
		converts_ = std::make_unique<compute::Converts>(fn_vect_, buffers_, stft_env_.gpu_stft_buffer_, gpu_3d_vision, desc, input.get_frame_desc());
		// Setting the cufft plans to work on the default stream.
		cufftSetStream(plan2d_, static_cast<cudaStream_t>(0));
		if (compute_desc_.compute_mode != Computation::Direct)
			update_n_requested_.exchange(true);
		refresh();
	}

	Pipe::~Pipe()
	{
		cudaFree(buffers_.gpu_float_buffer_);
		cudaFree(buffers_.gpu_output_buffer_);
		cudaFree(buffers_.gpu_input_buffer_);
	}

	bool Pipe::update_n_parameter(unsigned short n)
	{
		if (!ICompute::update_n_parameter(n))
			return false;
		/* gpu_input_buffer */
		cudaError_t error;
		cudaFree(buffers_.gpu_input_buffer_);

		/*We malloc 2 frames because we might need a second one if the vibrometry is enabled*/
		if ((error = cudaMalloc<cufftComplex>(&buffers_.gpu_input_buffer_,
			sizeof(cufftComplex) * (input_.get_pixels() * 2))) != CUDA_SUCCESS)
		{
			std::cerr << "Cuda error : " << cudaGetErrorString(error) << std::endl;
			return false;
		}
		notify_observers();
		return true;
	}

	void Pipe::request_queues()
	{
		if (request_3d_vision_.load())
		{
			camera::FrameDescriptor fd = output_.get_frame_desc();

			fd.depth = sizeof(float);
			gpu_3d_vision.reset(new Queue(fd, compute_desc_.nsamples.load(), "3DQueue"));
			request_3d_vision_.exchange(false);
		}

		if (request_delete_3d_vision_)
		{
			gpu_3d_vision.reset(nullptr);
			request_delete_3d_vision_.exchange(false);
		}

		if (request_stft_cuts_)
		{
			camera::FrameDescriptor fd_xz = output_.get_frame_desc();

			fd_xz.depth = (compute_desc_.img_type == ImgType::Complex) ?
				sizeof(cuComplex) : sizeof(ushort);
			uint buffer_depth = ((compute_desc_.img_type == ImgType::Complex) ? (sizeof(cufftComplex)) : (sizeof(float)));
			auto fd_yz = fd_xz;
			fd_xz.height = compute_desc_.nsamples;
			fd_yz.width = compute_desc_.nsamples;
			gpu_stft_slice_queue_xz.reset(new Queue(fd_xz, global::global_config.stft_cuts_output_buffer_size, "STFTCutXZ"));
			gpu_stft_slice_queue_yz.reset(new Queue(fd_yz, global::global_config.stft_cuts_output_buffer_size, "STFTCutYZ"));
			cudaMalloc(&buffers_.gpu_float_cut_xz_, fd_xz.frame_res() * buffer_depth);
			cudaMalloc(&buffers_.gpu_float_cut_yz_, fd_yz.frame_res() * buffer_depth);

			cudaMalloc(&buffers_.gpu_ushort_cut_xz_, fd_xz.frame_size());
			cudaMalloc(&buffers_.gpu_ushort_cut_yz_, fd_yz.frame_size());
			request_stft_cuts_.exchange(false);
		}

		if (request_delete_stft_cuts_)
		{
			cudaFree(buffers_.gpu_float_cut_xz_);
			cudaFree(buffers_.gpu_float_cut_yz_);
			cudaFree(buffers_.gpu_ushort_cut_xz_);
			cudaFree(buffers_.gpu_ushort_cut_yz_);

			gpu_stft_slice_queue_xz.reset();
			gpu_stft_slice_queue_yz.reset();
			request_delete_stft_cuts_.exchange(false);
		}
	}
	
	namespace
	{
		void enqueue_lens(Queue *queue, cuComplex *lens_buffer, const FrameDescriptor& input_fd)
		{
			if (queue)
			{
				cuComplex* copied_lens_ptr = static_cast<cuComplex*>(queue->get_end());
				queue->enqueue(lens_buffer);
				normalize_complex(copied_lens_ptr, input_fd.frame_res());
			}
		}
	}

	void Pipe::refresh()
	{
		if (compute_desc_.compute_mode == Computation::Direct)
		{
			fn_vect_.clear();
			update_n_requested_.exchange(false);
			refresh_requested_.exchange(false);
			return;
		}
		// Allocating filter2d/flowagrpahy/convolution buffers
		ICompute::refresh();

		/* As the Pipe uses a single CUDA stream for its computations,
		 * we have to explicitly use the default stream (0).
		 * Because std::bind does not allow optional parameters to be
		 * deduced and bound, we have to use static_cast<cudaStream_t>(0) systematically. */

		const camera::FrameDescriptor& input_fd = input_.get_frame_desc();
		const camera::FrameDescriptor& output_fd = output_.get_frame_desc();

		/* Clean current vector. */
		fn_vect_.clear();

		// Updating number of images
		if (update_n_requested_)
		{
			if (!update_n_parameter(compute_desc_.nsamples))
			{
				compute_desc_.pindex.exchange(0);
				compute_desc_.nsamples.exchange(1);
				update_n_parameter(1);
				std::cerr << "Updating #img failed, #img updated to 1" << std::endl;
			}
			update_n_requested_.exchange(false);
		}

		// Allocating cuts/3d vision queues
		request_queues();

		// Allocating accumulation queues/buffers
		if (update_acc_requested_)
		{
			FrameDescriptor new_fd_xz = input_.get_frame_desc();
			FrameDescriptor new_fd_yz = input_.get_frame_desc();
			new_fd_xz.height = compute_desc_.nsamples;
			new_fd_yz.width = compute_desc_.nsamples;
			update_acc_parameter(gpu_img_acc_yz_, compute_desc_.img_acc_slice_yz_enabled, compute_desc_.img_acc_slice_yz_level, new_fd_yz);
			update_acc_parameter(gpu_img_acc_xz_, compute_desc_.img_acc_slice_xz_enabled, compute_desc_.img_acc_slice_xz_level, new_fd_xz);
			update_acc_requested_.exchange(false);
		}

		// Allocating ref_diff queues and starting the counter
		if (update_ref_diff_requested_)
		{
			update_ref_diff_parameter();
			ref_diff_counter = compute_desc_.ref_diff_level;
			update_ref_diff_requested_ = false;
		}

		// Aborting if allocation failed
		if (abort_construct_requested_)
		{
			refresh_requested_ = false;
			return;
		}

		if (autofocus_requested_ && autofocus_->get_state() == compute::STOPPED)
		{
			autofocus_->insert_init();
			autofocus_requested_ = false;
			return;
		}

		// Converting input_ to complex.
		if (autofocus_->get_state() == compute::STOPPED)
			fn_vect_.push_back(std::bind(
				make_contiguous_complex,
				std::ref(input_),
				buffers_.gpu_input_buffer_,
				static_cast<cudaStream_t>(0)));
		else if (autofocus_->get_state() == compute::COPYING)
		{
			autofocus_->insert_copy();
			return;
		}

		detect_intensity_.insert_post_contiguous_complex();
		autofocus_->insert_restore();

		if (compute_desc_.interpolation_enabled)
			fn_vect_.push_back([=]() {
				const float ratio = compute_desc_.interp_lambda > 0 ? compute_desc_.lambda / compute_desc_.interp_lambda : 1;
				tex_interpolation(buffers_.gpu_input_buffer_,
					input_fd.width,
					input_fd.height,
					ratio); });

		if (compute_desc_.ref_diff_enabled)
			fn_vect_.push_back(std::bind(
				&Pipe::handle_reference,
				this,
				buffers_.gpu_input_buffer_,
				1));

		// Handling ref_sliding
		if (compute_desc_.ref_sliding_enabled)
			fn_vect_.push_back(std::bind(
				&Pipe::handle_sliding_reference,
				this,
				buffers_.gpu_input_buffer_,
				1));

		fourier_transforms_->insert_fft();
		fourier_transforms_->insert_stft();

		// Handling depth accumulation
		if (compute_desc_.p_accu_enabled)
				fn_vect_.push_back([=]() {
					int pmin = compute_desc_.pindex.load();
					int pmax = std::max(0,
						std::min(pmin + compute_desc_.p_acc_level, static_cast<int>(compute_desc_.nsamples)));
					stft_moment(
						stft_env_.gpu_stft_buffer_,
						buffers_.gpu_input_buffer_,
						input_fd.frame_res(),
						pmin,
						pmax,
						compute_desc_.nsamples.load());
				});

		// Handling image ratio Ip/Iq
		if (compute_desc_.vibrometry_enabled)
		{
			// pframe is at 'gpu_input_buffer[0]'
			// qframe is at 'gpu_input_buffer[1]'
			cufftComplex* qframe = buffers_.gpu_input_buffer_ + input_fd.frame_res();
			fn_vect_.push_back(std::bind(
				frame_ratio,
				buffers_.gpu_input_buffer_,
				qframe,
				buffers_.gpu_input_buffer_,
				input_fd.frame_res(),
				static_cast<cudaStream_t>(0)));
		}

		// Handling convolution
		if (compute_desc_.convolution_enabled)
		{
			gpu_special_queue_start_index = 0;
			gpu_special_queue_max_index = compute_desc_.special_buffer_size;
			fn_vect_.push_back(std::bind(
				convolution_kernel,
				buffers_.gpu_input_buffer_,
				gpu_special_queue_,
				input_fd.frame_res(),
				input_fd.width,
				gpu_kernel_buffer_,
				compute_desc_.convo_matrix_width.load(),
				compute_desc_.convo_matrix_height.load(),
				compute_desc_.convo_matrix_z.load(),
				gpu_special_queue_start_index,
				gpu_special_queue_max_index,
				static_cast<cudaStream_t>(0)));
		}

		// Handling flowgraphy
		if (compute_desc_.flowgraphy_enabled)
		{
			gpu_special_queue_start_index = 0;
			gpu_special_queue_max_index = compute_desc_.special_buffer_size.load();
			fn_vect_.push_back(std::bind(
				convolution_flowgraphy,
				buffers_.gpu_input_buffer_,
				gpu_special_queue_,
				std::ref(gpu_special_queue_start_index),
				gpu_special_queue_max_index,
				input_fd.frame_res(),
				input_fd.width,
				compute_desc_.flowgraphy_level.load(),
				static_cast<cudaStream_t>(0)));
		}

		// handling complex recording
		if (complex_output_requested_)
		{
			fn_vect_.push_back(std::bind(
				&Pipe::record_complex,
				this,
				buffers_.gpu_input_buffer_,
				static_cast<cudaStream_t>(0)));
		}

		converts_->insert_to_float();
		if (compute_desc_.img_type == ImgType::Argument)
		{
			fn_vect_.push_back(std::bind(
				complex_to_argument,
				buffers_.gpu_input_buffer_,
				buffers_.gpu_float_buffer_,
				input_fd.frame_res(),
				static_cast<cudaStream_t>(0)));

			if (unwrap_2d_requested_.load())
			{
				try
				{
					if (!unwrap_res_2d_)
						unwrap_res_2d_.reset(new UnwrappingResources_2d(input_.get_pixels()));
					if (unwrap_res_2d_->image_resolution_ != input_.get_pixels())
						unwrap_res_2d_->reallocate(input_.get_pixels());

					fn_vect_.push_back(std::bind(
						unwrap_2d,
						buffers_.gpu_float_buffer_,
						plan2d_,
						unwrap_res_2d_.get(),
						input_.get_frame_desc(),
						unwrap_res_2d_->gpu_angle_,
						static_cast<cudaStream_t>(0)));

					// Converting angle information in floating-point representation.
					fn_vect_.push_back(std::bind(
						rescale_float_unwrap2d,
						unwrap_res_2d_->gpu_angle_,
						buffers_.gpu_float_buffer_,
						unwrap_res_2d_->minmax_buffer_,
						input_fd.frame_res(),
						static_cast<cudaStream_t>(0)));
				}
				catch (std::exception& e)
				{
					std::cout << e.what() << std::endl;
				}
			}
			else
			{
				// Converting angle information in floating-point representation.
				fn_vect_.push_back(std::bind(
					rescale_argument,
					buffers_.gpu_float_buffer_,
					input_fd.frame_res(),
					static_cast<cudaStream_t>(0)));
			}
		}
		else if (compute_desc_.img_type == ImgType::PhaseIncrease)
		{
			//Unwrap_res is a ressource for phase_increase
			try
			{
				if (!unwrap_res_)
					unwrap_res_.reset(new UnwrappingResources(
						compute_desc_.unwrap_history_size,
						input_.get_pixels()));
				unwrap_res_->reset(compute_desc_.unwrap_history_size);
				unwrap_res_->reallocate(input_.get_pixels());
				if (compute_desc_.img_type.load() == ImgType::PhaseIncrease)
					fn_vect_.push_back(std::bind(
						phase_increase,
						buffers_.gpu_input_buffer_,
						unwrap_res_.get(),
						input_fd.frame_res()));
				else
					// Fallback on modulus mode.
					fn_vect_.push_back(std::bind(
						complex_to_modulus,
						buffers_.gpu_input_buffer_,
						buffers_.gpu_float_buffer_,
						input_fd.frame_res(),
						static_cast<cudaStream_t>(0)));

				if (unwrap_2d_requested_)
				{
					if (!unwrap_res_2d_)
						unwrap_res_2d_.reset(new UnwrappingResources_2d(input_.get_pixels()));

					if (unwrap_res_2d_->image_resolution_ != input_.get_pixels())
						unwrap_res_2d_->reallocate(input_.get_pixels());

					fn_vect_.push_back(std::bind(
						unwrap_2d,
						unwrap_res_->gpu_angle_current_,
						plan2d_,
						unwrap_res_2d_.get(),
						input_.get_frame_desc(),
						unwrap_res_2d_->gpu_angle_,
						static_cast<cudaStream_t>(0)));

					// Converting angle information in floating-point representation.
					fn_vect_.push_back(std::bind(
						rescale_float_unwrap2d,
						unwrap_res_2d_->gpu_angle_,
						buffers_.gpu_float_buffer_,
						unwrap_res_2d_->minmax_buffer_,
						input_fd.frame_res(),
						static_cast<cudaStream_t>(0)));
				}
				else
					// Converting angle information in floating-point representation.
					fn_vect_.push_back(std::bind(
						rescale_float,
						unwrap_res_->gpu_angle_current_,
						buffers_.gpu_float_buffer_,
						input_fd.frame_res(),
						static_cast<cudaStream_t>(0)));
			}
			catch (std::exception& e)
			{
				std::cout << e.what() << std::endl;
			}
		}
		else if (compute_desc_.img_type == Complex)
		{
			refresh_requested_ = false;
			autocontrast_requested_ = false;
			return;
		}

		stabilization_->insert_post_img_type();

		// Inserts the output buffers into the accumulation queues
		// and rewrite the average into the output buffer
		// For the XY view, this happens in stabilization.cc,
		// as the average is used in the intermediate computations
		if (compute_desc_.img_acc_slice_yz_enabled)
			enqueue_buffer(gpu_img_acc_yz_.get(),
				buffers_.gpu_float_cut_yz_,
				compute_desc_.img_acc_slice_yz_level,
				input_fd.height * compute_desc_.nsamples);
		if (compute_desc_.img_acc_slice_xz_enabled)
			enqueue_buffer(gpu_img_acc_xz_.get(),
				buffers_.gpu_float_cut_xz_,
				compute_desc_.img_acc_slice_xz_level,
				input_fd.width * compute_desc_.nsamples);



		contrast_->insert_fft_shift();

		// Handling noise/signal average
		if (average_requested_)
		{
			units::RectFd signalZone;
			units::RectFd noiseZone;
			compute_desc_.signalZone(signalZone, AccessMode::Get);
			compute_desc_.noiseZone(noiseZone, AccessMode::Get);
			// Recording or diplaying. TODO: allow both at the same time
			if (average_record_requested_)
			{
				fn_vect_.push_back(std::bind(
					&Pipe::average_record_caller,
					this,
					buffers_.gpu_float_buffer_,
					input_fd.width,
					input_fd.height,
					signalZone,
					noiseZone,
					static_cast<cudaStream_t>(0)));

				average_record_requested_ = false;
			}
			else
				fn_vect_.push_back(std::bind(
					&Pipe::average_caller,
					this,
					buffers_.gpu_float_buffer_,
					input_fd.width,
					input_fd.height,
					signalZone,
					noiseZone,
					static_cast<cudaStream_t>(0)));
		}

		contrast_->insert_log();
		contrast_->insert_contrast();
		autofocus_->insert_autofocus();

		// Handling float recording
		if (float_output_requested_)
		{
			fn_vect_.push_back(std::bind(
				&Pipe::record_float,
				this,
				buffers_.gpu_float_buffer_,
				static_cast<cudaStream_t>(0)));
		}

		// Displaying fps
		fn_vect_.push_back(std::bind(
			&Pipe::fps_count,
			this));

		// Converting float buffers to short output buffers
		if (!compute_desc_.vision_3d_enabled)
			fn_vect_.push_back(std::bind(
				float_to_ushort,
				buffers_.gpu_float_buffer_,
				buffers_.gpu_output_buffer_,
				buffers_.gpu_float_buffer_size_ / sizeof(float),
				output_fd.depth,
				static_cast<cudaStream_t>(0)));

		if (compute_desc_.stft_view_enabled)
		{
			fn_vect_.push_back(std::bind(
				float_to_ushort,
				buffers_.gpu_float_cut_xz_,
				buffers_.gpu_ushort_cut_xz_,
				get_stft_slice_queue(0).get_frame_desc().frame_res(),
				2.f, static_cast<cudaStream_t>(0)));
			fn_vect_.push_back(std::bind(
				float_to_ushort,
				buffers_.gpu_float_cut_yz_,
				buffers_.gpu_ushort_cut_yz_,
				get_stft_slice_queue(1).get_frame_desc().frame_res(),
				2.f, static_cast<cudaStream_t>(0)));
		}
		refresh_requested_.exchange(false);
	}

	void *Pipe::get_enqueue_buffer()
	{
		return compute_desc_.img_type.load() == ImgType::Complex ? buffers_.gpu_input_buffer_ : buffers_.gpu_output_buffer_;
	}

	void Pipe::exec()
	{
		if (global::global_config.flush_on_refresh)
			input_.flush();
		while (!termination_requested_.load())
		{
			if (input_.get_current_elts() >= 1)
			{
				stft_env_.stft_handle_ = false;
				for (FnType& f : fn_vect_)
				{
					f();
					if (stft_env_.stft_frame_counter_ != compute_desc_.stft_steps && stft_env_.stft_handle_)
						break;
				}
				if (compute_desc_.compute_mode == Hologram)
				{
					if (stft_env_.stft_frame_counter_ == compute_desc_.stft_steps)
					{
						if (!output_.enqueue(get_enqueue_buffer()))
						{
							input_.dequeue();
							break;
						}
						if (compute_desc_.stft_view_enabled)
						{
							queue_enqueue(compute_desc_.img_type == Complex ? buffers_.gpu_float_cut_xz_ : buffers_.gpu_ushort_cut_xz_,
								gpu_stft_slice_queue_xz.get());
							queue_enqueue(compute_desc_.img_type == Complex ? buffers_.gpu_float_cut_yz_ : buffers_.gpu_ushort_cut_yz_,
								gpu_stft_slice_queue_yz.get());
						}
					}
				}
				else if (!output_.enqueue(input_.get_start()))
				{
					input_.dequeue();
					break;
				}

				input_.dequeue();
				if (refresh_requested_.load())
					refresh();
			}
		}
	}

	void Pipe::enqueue_buffer(Queue* queue, float *buffer, uint nb_images, uint nb_pixels)
	{
		if (!queue)
		{
			std::cout << "Error: queue is null" << std::endl;
			return;
		}
		/*Add image to phase accumulation buffer*/

		fn_vect_.push_back([=]() {queue_enqueue(buffer, queue); } );
		fn_vect_.push_back(std::bind(
			accumulate_images,
			static_cast<float *>(queue->get_buffer()),
			buffer,
			queue->get_start_index(),
			queue->get_max_elts(),
			nb_images,
			nb_pixels,
			static_cast<cudaStream_t>(0)));
	}

	Queue* Pipe::get_lens_queue()
	{
		return fourier_transforms_->get_lens_queue();
	}
}