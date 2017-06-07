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
#include <algorithm>

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
#include "flowgraphy.cuh"
#include "tools.cuh"
#include "autofocus.cuh"
#include "tools_conversion.cuh"
#include "tools.hh"
#include "preprocessing.cuh"
#include "contrast_correction.cuh"
#include "vibrometry.cuh"

namespace holovibes
{
	Pipe::Pipe(
		Queue& input,
		Queue& output,
		ComputeDescriptor& desc)
		: ICompute(input, output, desc)
		, fn_vect_()
		, gpu_input_buffer_(nullptr)
		, gpu_output_buffer_(nullptr)
		, gpu_float_buffer_(nullptr)
		, gpu_input_frame_ptr_(nullptr)
	{
		int err = 0;
		int complex_pixels = sizeof(cufftComplex) * input_.get_pixels();
		camera::FrameDescriptor fd = output_.get_frame_desc();

		if (cudaMalloc<cufftComplex>(&gpu_input_buffer_, complex_pixels * input_length_) != cudaSuccess)
			err++;
		if (cudaMalloc(&gpu_output_buffer_, fd.depth * input_.get_pixels()) != cudaSuccess)
			err++;
		if (cudaMalloc<float>(&gpu_float_buffer_, sizeof(float) * input_.get_pixels()) != cudaSuccess)
			err++;
		if (err != 0)
			throw std::exception(cudaGetErrorString(cudaGetLastError()));
		
		// Setting the cufft plans to work on the default stream.
		cufftSetStream(plan1d_, static_cast<cudaStream_t>(0));
		cufftSetStream(plan2d_, static_cast<cudaStream_t>(0));
		cufftSetStream(plan3d_, static_cast<cudaStream_t>(0));

		refresh();
	}

	Pipe::~Pipe()
	{
		if (gpu_float_buffer_)
			cudaFree(gpu_float_buffer_);
		if (gpu_output_buffer_)
			cudaFree(gpu_output_buffer_);
		if (gpu_input_buffer_)
			cudaFree(gpu_input_buffer_);
	}

	bool Pipe::update_n_parameter(unsigned short n)
	{
		//const int p = compute_desc_.pindex.load();
		//compute_desc_.pindex.exchange(0);
		if (!ICompute::update_n_parameter(n))
			return (false);
		/* gpu_input_buffer */
		cudaError_t error;
		cudaFree(gpu_input_buffer_);

		if (compute_desc_.stft_enabled.load())
		{
			/*We malloc 2 frames because we might need a second one if the vibrometry is enabled*/
			if ((error = cudaMalloc<cufftComplex>(&gpu_input_buffer_,
				sizeof(cufftComplex) * (input_.get_pixels() * 2))) != CUDA_SUCCESS)
			{
				std::cerr << "Cuda error : " << cudaGetErrorString(error) << std::endl;
				return (false);
			}
		}
		else
		{
			if ((error = cudaMalloc<cufftComplex>(&gpu_input_buffer_,
				sizeof(cufftComplex) * input_.get_pixels() * input_length_)) != CUDA_SUCCESS)
			{
				std::cerr << "Cuda error : " << cudaGetErrorString(error) << std::endl;
				return (false);
			}
		}
		notify_observers();
		return (true);
	}

	void Pipe::request_queues()
	{
		if (request_3d_vision_.load())
		{
			camera::FrameDescriptor fd = output_.get_frame_desc();

			fd.depth = sizeof(float);
			gpu_3d_vision = new Queue(fd, compute_desc_.nsamples.load(), "3DQueue");
			request_3d_vision_.exchange(false);
		}

		if (request_delete_3d_vision_.load())
		{
			if (gpu_3d_vision)
			{
				delete gpu_3d_vision;
				gpu_3d_vision = nullptr;
			}
			request_delete_3d_vision_.exchange(false);
		}

		if (request_stft_cuts_.load())
		{
			camera::FrameDescriptor fd = output_.get_frame_desc();
			fd.height = compute_desc_.nsamples.load();

			fd.depth = (compute_desc_.view_mode == ComplexViewMode::Complex) ?
				sizeof(cuComplex) : sizeof(ushort);
			gpu_stft_slice_queue_xz.reset(new Queue(fd, global::global_config.stft_cuts_output_buffer_size, "STFTCutXZ"));
			gpu_stft_slice_queue_yz.reset(new Queue(fd, global::global_config.stft_cuts_output_buffer_size, "STFTCutYZ"));
			cudaMalloc(&gpu_float_cut_xz_, fd.frame_res() * ((compute_desc_.view_mode == ComplexViewMode::Complex) ? (sizeof(cufftComplex)) : (sizeof(float))));
			cudaMalloc(&gpu_float_cut_yz_, fd.frame_res() * ((compute_desc_.view_mode == ComplexViewMode::Complex) ? (sizeof(cufftComplex)) : (sizeof(float))));

			cudaMalloc(&gpu_ushort_cut_xz_, fd.frame_size());
			cudaMalloc(&gpu_ushort_cut_yz_, fd.frame_size());
			request_stft_cuts_.exchange(false);
		}

		if (request_delete_stft_cuts_.load())
		{
			if (gpu_float_cut_xz_)	cudaFree(gpu_float_cut_xz_);
			if (gpu_float_cut_yz_)	cudaFree(gpu_float_cut_yz_);
			if (gpu_ushort_cut_xz_)	cudaFree(gpu_ushort_cut_xz_);
			if (gpu_ushort_cut_yz_)	cudaFree(gpu_ushort_cut_yz_);

			if (gpu_stft_slice_queue_xz)
			{
				gpu_stft_slice_queue_xz.reset(nullptr);
			}
			if (gpu_stft_slice_queue_yz)
			{
				gpu_stft_slice_queue_yz.reset(nullptr);
			}
			request_delete_stft_cuts_.exchange(false);
		}
	}

	void Pipe::direct_refresh()
	{
		const camera::FrameDescriptor& input_fd = input_.get_frame_desc();
		const camera::FrameDescriptor& output_fd = output_.get_frame_desc();

		if (abort_construct_requested_.load())
		{
			refresh_requested_.exchange(false);
			return;
		}
		fn_vect_.push_back(std::bind(
			make_contiguous_complex,
			std::ref(input_),
			gpu_input_buffer_,
			input_length_,
			static_cast<cudaStream_t>(0)));
		gpu_input_frame_ptr_ = gpu_input_buffer_;
		fn_vect_.push_back(std::bind(
			complex_to_modulus,
			gpu_input_frame_ptr_,
			gpu_float_buffer_,
			input_fd.frame_res(),
			static_cast<cudaStream_t>(0)));
		fn_vect_.push_back(std::bind(
			float_to_ushort,
			gpu_float_buffer_,
			gpu_output_buffer_,
			input_fd.frame_res(),
			output_fd.depth,
			static_cast<cudaStream_t>(0)));
		refresh_requested_.exchange(false);
	}

	void Pipe::refresh()
	{
		if (compute_desc_.compute_mode.load() == Computation::Direct)
		{
			fn_vect_.clear();	
			update_n_requested_.exchange(false);
			direct_refresh();
			return;
		}
		if (compute_desc_.compute_mode.load() != Computation::Direct)
			ICompute::refresh();
		/* As the Pipe uses a single CUDA stream for its computations,
		 * we have to explicitly use the default stream (0).
		 * Because std::bind does not allow optional parameters to be
		 * deduced and bound, we have to use static_cast<cudaStream_t>(0) systematically. */

		const camera::FrameDescriptor& input_fd = input_.get_frame_desc();
		const camera::FrameDescriptor& output_fd = output_.get_frame_desc();

		//refresh_requested_.exchange(false);
		/* Clean current vector. */
		fn_vect_.clear();
		if (update_n_requested_.load())
		{
			if (!update_n_parameter(compute_desc_.nsamples.load()))
			{
				compute_desc_.pindex.exchange(0);
				compute_desc_.nsamples.exchange(1);
				update_n_parameter(1);
				std::cerr << "Updating #img failed, #img updated to 1" << std::endl;
			}
			update_n_requested_.exchange(false);
		}

		request_queues();

		if (update_acc_requested_.load())
		{
			update_acc_requested_.exchange(false);
			update_acc_parameter();
		}

		if (update_ref_diff_requested_.load())
		{
			update_ref_diff_parameter();
			ref_diff_counter = compute_desc_.ref_diff_level.load();
			update_ref_diff_requested_.exchange(false);
		}

		if (abort_construct_requested_.load())
		{
			refresh_requested_.exchange(false);
			return;
		}

		if (autofocus_requested_.load())
		{
			fn_vect_.push_back(std::bind(
				&Pipe::autofocus_caller,
				this,
				gpu_float_buffer_,
				static_cast<cudaStream_t>(0)));
			autofocus_requested_.exchange(false);
			request_refresh();
			return;
		}

		// Fill input complex buffer, one frame at a time.
		fn_vect_.push_back(std::bind(
			make_contiguous_complex,
			std::ref(input_),
			gpu_input_buffer_,
			input_length_,
			static_cast<cudaStream_t>(0)));

		unsigned int nframes = compute_desc_.nsamples.load();
		unsigned int pframe = compute_desc_.pindex.load();
		unsigned int qframe = compute_desc_.vibrometry_q.load();

		gui::Rectangle roiZone;
		compute_desc_.stftRoiZone(roiZone, AccessMode::Get);
		if (compute_desc_.stft_enabled.load() ||
			(compute_desc_.filter_2d_enabled.load() && !roiZone.area()))
		{
			nframes = 1;
			pframe = 0;
			qframe = 0;
		}

		// In this case the q frame is not needed and will therefore not be computed in FFT1 & FFT2 
		if (!compute_desc_.vibrometry_enabled.load())
			qframe = pframe;
		/* p frame pointer */
		gpu_input_frame_ptr_ = gpu_input_buffer_ + pframe * input_fd.frame_res();

		if (compute_desc_.ref_diff_enabled.load())
			fn_vect_.push_back(std::bind(
				&Pipe::handle_reference,
				this,
				gpu_input_buffer_,
				nframes));

		if (compute_desc_.ref_sliding_enabled.load())
			fn_vect_.push_back(std::bind(
				&Pipe::handle_sliding_reference,
				this,
				gpu_input_buffer_,
				nframes));

		if (compute_desc_.filter_2d_enabled.load())
		{
			fn_vect_.push_back(std::bind(
				filter2D,
				gpu_input_buffer_,
				gpu_filter2d_buffer,
				plan2d_,
				roiZone,
				input_fd,
				static_cast<cudaStream_t>(0)));
		}
		//Algorithm combobox
		if (!compute_desc_.filter_2d_enabled.load() ||
			(compute_desc_.filter_2d_enabled.load() && roiZone.area()))
		{
			if (compute_desc_.algorithm.load() == Algorithm::None)
			{
				// Add temporal FFT1 1D .
				fn_vect_.push_back(std::bind(
					demodulation,
					gpu_input_buffer_,
					plan1d_,
					static_cast<cudaStream_t>(0)));
			}
			else if (compute_desc_.algorithm.load() == Algorithm::FFT1)
			{
				fft1_lens(
					gpu_lens_,
					input_fd,
					compute_desc_.lambda.load(),
					compute_desc_.zdistance.load(),
					static_cast<cudaStream_t>(0));
				// Add FFT1.
				fn_vect_.push_back(std::bind(
					fft_1,
					gpu_input_buffer_,
					gpu_lens_,
					plan1d_,
					plan2d_,
					input_fd.frame_res(),
					nframes,
					pframe,
					qframe,
					static_cast<cudaStream_t>(0)));

				if (compute_desc_.vibrometry_enabled.load())
				{

					/* q frame pointer */
					cufftComplex* q = gpu_input_buffer_ + qframe * input_fd.frame_res();

					fn_vect_.push_back(std::bind(
						frame_ratio,
						gpu_input_frame_ptr_,
						q,
						gpu_input_frame_ptr_,
						input_fd.frame_res(),
						static_cast<cudaStream_t>(0)));
				}
			}
			else if (compute_desc_.algorithm.load() == Algorithm::FFT2)
			{
				fft2_lens(
					gpu_lens_,
					input_fd,
					compute_desc_.lambda.load(),
					compute_desc_.zdistance.load(),
					static_cast<cudaStream_t>(0));

				fn_vect_.push_back(std::bind(
					fft_2,
					gpu_input_buffer_,
					gpu_lens_,
					plan1d_,
					plan2d_,
					input_fd,
					nframes,
					pframe,
					qframe,
					static_cast<cudaStream_t>(0)));

				if (compute_desc_.vibrometry_enabled.load())
				{
					/* q frame pointer */
					cufftComplex* q = gpu_input_buffer_ + qframe * input_fd.frame_res();

					fn_vect_.push_back(std::bind(
						frame_ratio,
						gpu_input_frame_ptr_,
						q,
						gpu_input_frame_ptr_,
						input_fd.frame_res(),
						static_cast<cudaStream_t>(0)));
				}
			}
		}
		// STFT Checkbox
		if (compute_desc_.stft_enabled.load())
		{
			fn_vect_.push_back(std::bind(
				&ICompute::queue_enqueue,
				this,
				gpu_input_frame_ptr_,
				gpu_stft_queue_));

			fn_vect_.push_back(std::bind(
				&ICompute::stft_handler,
				this,
				gpu_input_buffer_,
				static_cast<cufftComplex *>(gpu_stft_queue_->get_buffer())));
			
			if (compute_desc_.p_accu_enabled.load())
			{
				fn_vect_.push_back(std::bind(stft_moment, gpu_stft_buffer_,
					gpu_input_frame_ptr_,
					input_fd.frame_res(),
					compute_desc_.p_accu_min_level.load(),
					compute_desc_.p_accu_max_level.load(),
					compute_desc_.nsamples.load()));
			}
			// Image ratio Ip/Iq chackbox
			if (compute_desc_.vibrometry_enabled.load())
			{
				qframe = 1;
				/* q frame pointer */
				cufftComplex* q = gpu_input_buffer_ + qframe * input_fd.frame_res();
				fn_vect_.push_back(std::bind(
					frame_ratio,
					gpu_input_buffer_,
					q,
					gpu_input_buffer_,
					input_fd.frame_res(),
					static_cast<cudaStream_t>(0)));
			}
			/* frame pointer */
			gpu_input_frame_ptr_ = gpu_input_buffer_;
		}

		if (compute_desc_.convolution_enabled.load())
		{
			gpu_special_queue_start_index = 0;
			gpu_special_queue_max_index = compute_desc_.special_buffer_size.load();
			fn_vect_.push_back(std::bind(
				convolution_kernel,
				gpu_input_frame_ptr_,
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

		if (compute_desc_.flowgraphy_enabled.load())
		{
			gpu_special_queue_start_index = 0;
			gpu_special_queue_max_index = compute_desc_.special_buffer_size.load();
			fn_vect_.push_back(std::bind(
				convolution_flowgraphy,
				gpu_input_frame_ptr_,
				gpu_special_queue_,
				std::ref(gpu_special_queue_start_index),
				gpu_special_queue_max_index,
				input_fd.frame_res(),
				input_fd.width,
				compute_desc_.flowgraphy_level.load(),
				static_cast<cudaStream_t>(0)));
		}

		if (complex_output_requested_.load())
		{
			fn_vect_.push_back(std::bind(
				&Pipe::record_complex,
				this,
				gpu_input_frame_ptr_,
				static_cast<cudaStream_t>(0)));
		}

		/* Apply conversion to floating-point respresentation. */
		if (compute_desc_.view_mode.load() == ComplexViewMode::Modulus)
		{
			if (compute_desc_.vision_3d_enabled.load())
				fn_vect_.push_back(std::bind(
					complex_to_modulus,
					gpu_stft_buffer_,
					reinterpret_cast<float *>(gpu_3d_vision->get_buffer()),
					input_fd.frame_res() * compute_desc_.nsamples.load(),
					static_cast<cudaStream_t>(0)));
			else
				fn_vect_.push_back(std::bind(
					complex_to_modulus,
					gpu_input_frame_ptr_,
					gpu_float_buffer_,
					input_fd.frame_res(),
					static_cast<cudaStream_t>(0)));
		}
		else if (compute_desc_.view_mode.load() == ComplexViewMode::SquaredModulus)
		{
			fn_vect_.push_back(std::bind(
				complex_to_squared_modulus,
				gpu_input_frame_ptr_,
				gpu_float_buffer_,
				input_fd.frame_res(),
				static_cast<cudaStream_t>(0)));
		}
		else if (compute_desc_.view_mode.load() == ComplexViewMode::Argument)
		{
			fn_vect_.push_back(std::bind(
				complex_to_argument,
				gpu_input_frame_ptr_,
				gpu_float_buffer_,
				input_fd.frame_res(),
				static_cast<cudaStream_t>(0)));

			if (unwrap_2d_requested_.load())
			{
				try
				{
					if (!unwrap_res_2d_)
					{
						unwrap_res_2d_.reset(new UnwrappingResources_2d(
							input_.get_pixels()));
					}
					if (unwrap_res_2d_->image_resolution_ != input_.get_pixels())
						unwrap_res_2d_->reallocate(input_.get_pixels());

					fn_vect_.push_back(std::bind(
						unwrap_2d,
						gpu_float_buffer_,
						plan2d_,
						unwrap_res_2d_.get(),
						input_.get_frame_desc(),
						unwrap_res_2d_->gpu_angle_,
						static_cast<cudaStream_t>(0)));

					// Converting angle information in floating-point representation.
					fn_vect_.push_back(std::bind(
						rescale_float_unwrap2d,
						unwrap_res_2d_->gpu_angle_,
						gpu_float_buffer_,
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
					gpu_float_buffer_,
					input_fd.frame_res(),
					static_cast<cudaStream_t>(0)));
			}
		}
		else if (compute_desc_.view_mode.load() == ComplexViewMode::Complex)
		{
			fn_vect_.push_back(std::bind(
				complex_to_complex,
				gpu_input_frame_ptr_,
				reinterpret_cast<ushort *>(gpu_output_buffer_),
				input_fd.frame_res() << 3, // frame_res() * 8
				static_cast<cudaStream_t>(0)));
			refresh_requested_.exchange(false);
			if (autocontrast_requested_.load())
				autocontrast_requested_.exchange(false);
			return;
		}
		else
		{
			//Unwrap_res is a ressource for phase_increase
			try
			{
				if (!unwrap_res_)
				{
					unwrap_res_.reset(new UnwrappingResources(
						compute_desc_.unwrap_history_size.load(),
						input_.get_pixels()));
				}
				unwrap_res_->reset(compute_desc_.unwrap_history_size.load());
				unwrap_res_->reallocate(input_.get_pixels());
				if (compute_desc_.view_mode.load() == ComplexViewMode::PhaseIncrease)
				{
					// Phase increase
					fn_vect_.push_back(std::bind(
						phase_increase,
						gpu_input_frame_ptr_,
						unwrap_res_.get(),
						input_fd.frame_res()));
				}
				else
				{
					// Fallback on modulus mode.

					fn_vect_.push_back(std::bind(
						complex_to_modulus,
						gpu_input_frame_ptr_,
						gpu_float_buffer_,
						input_fd.frame_res(),
						static_cast<cudaStream_t>(0)));
				};

				if (unwrap_2d_requested_.load())
				{
					if (!unwrap_res_2d_)
					{
						unwrap_res_2d_.reset(new UnwrappingResources_2d(
							input_.get_pixels()));
					}
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
						gpu_float_buffer_,
						unwrap_res_2d_->minmax_buffer_,
						input_fd.frame_res(),
						static_cast<cudaStream_t>(0)));
				}
				else
				{
					// Converting angle information in floating-point representation.
					fn_vect_.push_back(std::bind(
						rescale_float,
						unwrap_res_->gpu_angle_current_,
						gpu_float_buffer_,
						input_fd.frame_res(),
						static_cast<cudaStream_t>(0)));
				}
			}
			catch (std::exception& e)
			{
				std::cout << e.what() << std::endl;
			}
		}


		/*Compute Accumulation buffer into gpu_float_buffer*/
		if (compute_desc_.img_acc_enabled.load())
		{
			/*Add image to phase accumulation buffer*/

			fn_vect_.push_back(std::bind(
				&ICompute::queue_enqueue,
				this,
				gpu_float_buffer_,
				gpu_img_acc_));

			fn_vect_.push_back(std::bind(
				accumulate_images,
				static_cast<float *>(gpu_img_acc_->get_buffer()),
				gpu_float_buffer_,
				gpu_img_acc_->get_start_index(),
				gpu_img_acc_->get_max_elts(),
				compute_desc_.img_acc_level.load(),
				input_fd.frame_res(),
				static_cast<cudaStream_t>(0)));
		}
		/* [POSTPROCESSING] Everything behind this line uses output_frame_ptr */

		if (compute_desc_.shift_corners_enabled.load())
		{
			fn_vect_.push_back(std::bind(
				shift_corners,
				gpu_float_buffer_,
				output_fd.width,
				output_fd.height,
				static_cast<cudaStream_t>(0)));
		}

		if (average_requested_.load())
		{
			gui::Rectangle signalZone;
			gui::Rectangle noiseZone;
			compute_desc_.signalZone(signalZone, AccessMode::Get);
			compute_desc_.noiseZone(noiseZone, AccessMode::Get);
			if (average_record_requested_.load())
			{
				fn_vect_.push_back(std::bind(
					&Pipe::average_record_caller,
					this,
					gpu_float_buffer_,
					input_fd.width,
					input_fd.height,
					signalZone,
					noiseZone,
					static_cast<cudaStream_t>(0)));

				average_record_requested_.exchange(false);
			}
			else
			{
				fn_vect_.push_back(std::bind(
					&Pipe::average_caller,
					this,
					gpu_float_buffer_,
					input_fd.width,
					input_fd.height,
					signalZone,
					noiseZone,
					static_cast<cudaStream_t>(0)));
			}
		}

		if (compute_desc_.log_scale_enabled.load()
			|| compute_desc_.log_scale_enabled_cut_xz.load()
			|| compute_desc_.log_scale_enabled_cut_yz.load())
		{
			if (compute_desc_.log_scale_enabled.load())
				fn_vect_.push_back(std::bind(
					apply_log10,
					gpu_float_buffer_,
					input_fd.frame_res(),
					static_cast<cudaStream_t>(0)));
			if (compute_desc_.stft_view_enabled.load())
			{
				if (compute_desc_.log_scale_enabled_cut_xz.load())
					fn_vect_.push_back(std::bind(
						apply_log10,
						static_cast<float *>(gpu_float_cut_xz_),
						output_fd.width * compute_desc_.nsamples.load(),
						static_cast<cudaStream_t>(0)));
				if (compute_desc_.log_scale_enabled_cut_yz.load())
					fn_vect_.push_back(std::bind(
						apply_log10,
						static_cast<float *>(gpu_float_cut_yz_),
						output_fd.height * compute_desc_.nsamples.load(),
						static_cast<cudaStream_t>(0)));
			}
		}

		/* ***************** */
		if (autocontrast_requested_.load())
		{
			if (compute_desc_.current_window.load() == WindowKind::MainDisplay)
			{
				if (compute_desc_.vision_3d_enabled.load())
					fn_vect_.push_back(std::bind(
						autocontrast_caller,
						reinterpret_cast<float *>(gpu_3d_vision->get_buffer()) + gpu_3d_vision->get_pixels() * compute_desc_.pindex.load(),
						output_fd.frame_res() * compute_desc_.nsamples.load(),
						0,
						std::ref(compute_desc_),
						std::ref(compute_desc_.contrast_min),
						std::ref(compute_desc_.contrast_max),
						static_cast<cudaStream_t>(0)));
				else
					fn_vect_.push_back(std::bind(
						autocontrast_caller,
						gpu_float_buffer_,
						output_fd.frame_res(),
						0,
						std::ref(compute_desc_),
						std::ref(compute_desc_.contrast_min),
						std::ref(compute_desc_.contrast_max),
						static_cast<cudaStream_t>(0)));
			}
			if (compute_desc_.stft_view_enabled.load())
			{
				if (compute_desc_.current_window.load() == WindowKind::SliceXZ)
					fn_vect_.push_back(std::bind(
						autocontrast_caller,
						static_cast<float *>(gpu_float_cut_xz_),
						static_cast<uint>(output_fd.width * compute_desc_.nsamples.load()),
						static_cast<uint>(output_fd.width * compute_desc_.cuts_contrast_p_offset.load()),
						std::ref(compute_desc_),
						std::ref(compute_desc_.contrast_min_slice_xz),
						std::ref(compute_desc_.contrast_max_slice_xz),
						static_cast<cudaStream_t>(0)));
				else if (compute_desc_.current_window.load() == WindowKind::SliceYZ)
					fn_vect_.push_back(std::bind(
						autocontrast_caller,
						static_cast<float *>(gpu_float_cut_yz_),
						output_fd.width * compute_desc_.nsamples.load(),
						output_fd.width * compute_desc_.cuts_contrast_p_offset.load(),
						std::ref(compute_desc_),
						std::ref(compute_desc_.contrast_min_slice_yz),
						std::ref(compute_desc_.contrast_max_slice_yz),
						static_cast<cudaStream_t>(0)));
			}
			autocontrast_requested_.exchange(false);
			request_refresh();
		}
		/* ***************** */
		if (compute_desc_.contrast_enabled.load())
		{
			if (compute_desc_.vision_3d_enabled.load())
				fn_vect_.push_back(std::bind(
					manual_contrast_correction,
					reinterpret_cast<float *>(gpu_3d_vision->get_buffer()) + gpu_3d_vision->get_pixels() * compute_desc_.pindex.load(),
					output_fd.frame_res() * compute_desc_.nsamples.load(),
					65535,
					compute_desc_.contrast_min.load(),
					compute_desc_.contrast_max.load(),
					static_cast<cudaStream_t>(0)));
			else
				fn_vect_.push_back(std::bind(
					manual_contrast_correction,
					gpu_float_buffer_,
					output_fd.frame_res(),
					65535,
					compute_desc_.contrast_min.load(),
					compute_desc_.contrast_max.load(),
					static_cast<cudaStream_t>(0)));
			if (compute_desc_.stft_view_enabled.load())
			{
				fn_vect_.push_back(std::bind(
					manual_contrast_correction,
					static_cast<float *>(gpu_float_cut_xz_),
					output_fd.width * compute_desc_.nsamples.load(),
					65535,
					compute_desc_.contrast_min_slice_xz.load(),
					compute_desc_.contrast_max_slice_xz.load(),
					static_cast<cudaStream_t>(0)));
				fn_vect_.push_back(std::bind(
					manual_contrast_correction,
					static_cast<float *>(gpu_float_cut_yz_),
					output_fd.width * compute_desc_.nsamples.load(),
					65535,
					compute_desc_.contrast_min_slice_yz.load(),
					compute_desc_.contrast_max_slice_yz.load(),
					static_cast<cudaStream_t>(0)));
			}
		}

		if (float_output_requested_.load())
		{
			fn_vect_.push_back(std::bind(
				&Pipe::record_float,
				this,
				gpu_float_buffer_,
				static_cast<cudaStream_t>(0)));
		}

		if (gui::InfoManager::get_manager())
			fn_vect_.push_back(std::bind(
				&Pipe::fps_count,
				this));

		if (!compute_desc_.vision_3d_enabled.load())
			fn_vect_.push_back(std::bind(
				float_to_ushort,
				gpu_float_buffer_,
				gpu_output_buffer_,
				input_fd.frame_res(),
				output_fd.depth,
				static_cast<cudaStream_t>(0)));

		if (compute_desc_.stft_view_enabled.load())
		{
			fn_vect_.push_back(std::bind(
				float_to_ushort,
				reinterpret_cast<float *>(gpu_float_cut_xz_),
				gpu_ushort_cut_xz_,
				get_stft_slice_queue(0).get_frame_desc().frame_res(),
				2.f, static_cast<cudaStream_t>(0)));
			fn_vect_.push_back(std::bind(
				float_to_ushort,
				reinterpret_cast<float *>(gpu_float_cut_yz_),
				gpu_ushort_cut_yz_,
				get_stft_slice_queue(1).get_frame_desc().frame_res(),
				2.f, static_cast<cudaStream_t>(0)));
		}
		refresh_requested_.exchange(false);
	}

	void Pipe::autofocus_caller(float* input, cudaStream_t stream)
	{
		/* Fill gpu_input complex buffer. */
		int err = 0;
		make_contiguous_complex(
			input_,
			gpu_input_buffer_,
			compute_desc_.nsamples.load());

		float z_min = compute_desc_.autofocus_z_min.load();
		float z_max = compute_desc_.autofocus_z_max.load();
		const float z_div = static_cast<float>(compute_desc_.autofocus_z_div.load());
		gui::Rectangle zone;
		compute_desc_.autofocusZone(zone, AccessMode::Get);

		/* Autofocus needs to work on the same images.
		* It will computes on copies. */
		cufftComplex* gpu_input_buffer_tmp;
		const size_t gpu_input_buffer_size = input_.get_pixels() * compute_desc_.nsamples.load() * sizeof(cufftComplex);
		if (cudaMalloc(&gpu_input_buffer_tmp, gpu_input_buffer_size) != cudaSuccess)
			err++;
		float z_step = (z_max - z_min) / z_div;

		/* Compute square af zone. */
		float* gpu_float_buffer_af_zone;
		const unsigned int zone_width = zone.width();
		const unsigned int zone_height = zone.height();

		const unsigned int af_square_size =
			static_cast<unsigned int>(powf(2, ceilf(log2f(zone_width > zone_height ?
				static_cast<float>(zone_width) : static_cast<float>(zone_height)))));
		const unsigned int af_size = af_square_size * af_square_size;

		if (cudaMalloc(&gpu_float_buffer_af_zone, af_size * sizeof(float)) == cudaSuccess)
			cudaMemset(gpu_float_buffer_af_zone, 0, af_size * sizeof(float));
		else
			err++;

		/// The main loop that calculates all z, and find the max one
		// z_step will decrease and zmin and zmax will merge into
		// the best autofocus_value.
		float af_z = 0.0f;

		std::vector<float> focus_metric_values;
		auto biggest = focus_metric_values.begin();

		const camera::FrameDescriptor& input_fd = input_.get_frame_desc();
		const camera::FrameDescriptor& output_fd = output_.get_frame_desc();

		unsigned int max_pos = 0;
		const unsigned int z_iter = compute_desc_.autofocus_z_iter.load();
		if (err != 0)
		{
			std::cerr << "Autofocus : cannot allocate gpu memory" << std::endl;
			return;
		}
		for (unsigned i = 0; i < z_iter; ++i)
		{
			for (float z = z_min; !autofocus_stop_requested_.load() && z < z_max; z += z_step)
			{
				/* Make input frames copies. */
				cudaMemcpy(
					gpu_input_buffer_tmp,
					gpu_input_buffer_,
					gpu_input_buffer_size,
					cudaMemcpyDeviceToDevice);

				if (compute_desc_.algorithm.load() == Algorithm::FFT1)
				{
					fft1_lens(
						gpu_lens_,
						input_fd,
						compute_desc_.lambda.load(),
						z);

					fft_1(
						gpu_input_buffer_tmp,
						gpu_lens_,
						plan1d_,
						plan2d_,
						input_fd.frame_res(),
						compute_desc_.nsamples.load(),
						compute_desc_.pindex.load(),
						compute_desc_.pindex.load());

					gpu_input_frame_ptr_ = gpu_input_buffer_tmp + compute_desc_.pindex.load() * input_fd.frame_res();
				}
				else if (compute_desc_.algorithm.load() == Algorithm::FFT2)
				{
					fft2_lens(
						gpu_lens_,
						input_fd,
						compute_desc_.lambda.load(),
						z);

					gpu_input_frame_ptr_ = gpu_input_buffer_tmp + compute_desc_.pindex.load() * input_fd.frame_res();
					
					fft_2(
						gpu_input_buffer_tmp,
						gpu_lens_,
						plan1d_,
						plan2d_,
						input_fd,
						compute_desc_.nsamples.load(),
						compute_desc_.pindex.load(),
						compute_desc_.pindex.load());
				}
				else
					assert(!"Impossible case");

				if (compute_desc_.view_mode.load() == ComplexViewMode::Modulus)
				{
					complex_to_modulus(gpu_input_frame_ptr_, gpu_float_buffer_, input_fd.frame_res());
				}
				else if (compute_desc_.view_mode.load() == ComplexViewMode::SquaredModulus)
				{
					complex_to_squared_modulus(gpu_input_frame_ptr_, gpu_float_buffer_, input_fd.frame_res());
				}
				else if (compute_desc_.view_mode.load() == ComplexViewMode::Argument)
				{
					complex_to_argument(gpu_input_frame_ptr_, gpu_float_buffer_, input_fd.frame_res());
				}
				else
				{
					cudaFree(gpu_float_buffer_af_zone);
					cudaFree(gpu_input_buffer_tmp);
					return;
				}

				if (compute_desc_.shift_corners_enabled.load())
				{
					shift_corners(
						gpu_float_buffer_,
						output_.get_frame_desc().width,
						output_.get_frame_desc().height);
				}

				if (compute_desc_.contrast_enabled.load())
				{
					manual_contrast_correction(
						gpu_float_buffer_,
						input_fd.frame_res(),
						65535,
						compute_desc_.contrast_min.load(),
						compute_desc_.contrast_max.load());
				}

				float_to_ushort(gpu_float_buffer_, gpu_output_buffer_, input_fd.frame_res(), output_fd.depth);
				output_.enqueue(gpu_output_buffer_, cudaMemcpyDeviceToDevice);

				frame_memcpy(gpu_float_buffer_, zone, input_fd.width, gpu_float_buffer_af_zone, af_square_size);

				const float focus_metric_value = focus_metric(gpu_float_buffer_af_zone, af_square_size, stream, compute_desc_.autofocus_size.load());

				if (!std::isnan(focus_metric_value))
					focus_metric_values.push_back(focus_metric_value);
				else
					focus_metric_values.push_back(0);
			}
			/* Find max z */
			biggest = std::max_element(focus_metric_values.begin(), focus_metric_values.end());

			/* Case the max has not been found. */
			if (biggest == focus_metric_values.end())
				biggest = focus_metric_values.begin();
			max_pos = std::distance(focus_metric_values.begin(), biggest);

			// This is our temp max
			af_z = z_min + max_pos * z_step;

			// Calculation of the new max/min, taking the old step
			z_min = af_z - z_step;
			z_max = af_z + z_step;

			z_step = (z_max - z_min) / z_div;
			focus_metric_values.clear();
		}
		gui::InfoManager::get_manager()->remove_info("Status");
		/// End of the loop, free resources and notify the new z
		// Sometimes a value outside the initial upper and lower bounds can be found
		// Thus checking if af_z is within initial bounds
		if (af_z != 0 && af_z >= compute_desc_.autofocus_z_min.load() && af_z <= compute_desc_.autofocus_z_max.load())
			compute_desc_.zdistance.exchange(af_z);
		compute_desc_.notify_observers();

		if (gpu_float_buffer_af_zone)
			cudaFree(gpu_float_buffer_af_zone);
		if (gpu_input_buffer_tmp)
			cudaFree(gpu_input_buffer_tmp);
	}

	void *Pipe::get_enqueue_buffer()
	{
		if (compute_desc_.view_mode.load() != ComplexViewMode::Complex)
			return (gpu_output_buffer_);
		return (gpu_input_frame_ptr_);
	}

	void Pipe::exec()
	{
		if (global::global_config.flush_on_refresh)
			input_.flush();
		while (!termination_requested_.load())
		{
			if (input_.get_current_elts() >= input_length_)
			{
				for (FnType& f : fn_vect_) f();
				if (!output_.enqueue(
					get_enqueue_buffer(),
					cudaMemcpyDeviceToDevice))
				{
					input_.dequeue();
					break;
				}
				if (compute_desc_.view_mode == ComplexViewMode::Complex && compute_desc_.stft_view_enabled.load())
				{
					gpu_stft_slice_queue_xz->enqueue(
						gpu_float_cut_xz_,
						cudaMemcpyDeviceToDevice);
					gpu_stft_slice_queue_yz->enqueue(
						gpu_float_cut_yz_,
						cudaMemcpyDeviceToDevice);
				}
				else if (compute_desc_.stft_view_enabled.load())
				{
					gpu_stft_slice_queue_xz->enqueue(
						gpu_ushort_cut_xz_,
						cudaMemcpyDeviceToDevice);
					gpu_stft_slice_queue_yz->enqueue(
						gpu_ushort_cut_yz_,
						cudaMemcpyDeviceToDevice);
				}
				input_.dequeue();
				if (refresh_requested_.load())
					refresh();
			}
		}
	}
}