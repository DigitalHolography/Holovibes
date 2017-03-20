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

#include <algorithm>
#include <cassert>

#include "pipeline.hh"
#include "config.hh"
#include "info_manager.hh"
#include "compute_descriptor.hh"
#include "module.hh"
#include "queue.hh"
#include "tools.hh"

#include "fft1.cuh"
#include "fft2.cuh"
#include "stft.cuh"
#include "tools.cuh"
#include "tools_conversion.cuh"
#include "preprocessing.cuh"
#include "contrast_correction.cuh"
#include "vibrometry.cuh"
#include "autofocus.cuh"

namespace holovibes
{
	Pipeline::Pipeline(
		Queue& input,
		Queue& output,
		ComputeDescriptor& desc)
		: ICompute(input, output, desc)
		, step_count_before_refresh_(0)
	{
		float                         *gpu_float_buffer = nullptr;

		/* The 16-bit buffer is allocated (and deallocated) separately,
		 as no Module is associated directly to it. */
		cudaMalloc<unsigned short>(&gpu_short_buffer_,
			sizeof(unsigned short)* input_.get_pixels());

		cudaMalloc(&gpu_float_buffer, sizeof(float)* input_.get_pixels());
		gpu_float_buffers_.push_back(gpu_float_buffer);
		cudaMalloc(&gpu_float_buffer, sizeof(float)* input_.get_pixels());
		gpu_float_buffers_.push_back(gpu_float_buffer);

		update_n_parameter(compute_desc_.nsamples.load());
		refresh();
	}

	Pipeline::~Pipeline()
	{
		stop_pipeline();
		cudaFree(gpu_short_buffer_);
		delete_them(gpu_complex_buffers_, [](cufftComplex* buffer) { cudaFree(buffer); });
		delete_them(gpu_float_buffers_, [](float* buffer) { cudaFree(buffer); });
	}

	void Pipeline::stop_pipeline()
	{
		delete_them(modules_, [](Module* module) { delete module; });
	}

	void Pipeline::exec()
	{
		if (global::global_config.flush_on_refresh)
			input_.flush();
		while (!termination_requested_)
		{
			if (input_.get_current_elts() >= compute_desc_.nsamples.load())
			{
				// Say to each Module that there is work to be done.
				std::for_each(modules_.begin(),
					modules_.end(),
					[](Module* module) { module->task_done_ = false; });

				while (std::any_of(modules_.begin(),
					modules_.end(),
					[](Module *module) { return !module->task_done_; }))
				{
					continue;
				}

				// Now that everyone is finished, rotate datasets as seen by the Modules.
				step_forward();

				output_.enqueue(
					gpu_short_buffer_,
					cudaMemcpyDeviceToDevice);
				input_.dequeue();

				if (refresh_requested_
					&& (step_count_before_refresh_ == 0 || --step_count_before_refresh_ == 0))
				{
					refresh();
				}
			}
		}
	}

	bool Pipeline::update_n_parameter(unsigned short n)
	{
		ICompute::update_n_parameter(n);

		cufftComplex                  *gpu_complex_buffer = nullptr;

		delete_them(gpu_complex_buffers_, [](cufftComplex* buffer) { cudaFree(buffer); });
		if (cudaMalloc(&gpu_complex_buffer, sizeof(cufftComplex)* input_.get_pixels() * input_length_) != CUDA_SUCCESS)
			return (false);
		gpu_complex_buffers_.push_back(gpu_complex_buffer);
		if (cudaMalloc(&gpu_complex_buffer, sizeof(cufftComplex)* input_.get_pixels() * input_length_) != CUDA_SUCCESS)
			return (false); 
		gpu_complex_buffers_.push_back(gpu_complex_buffer);

		/* Remember that we don't need to deallocate these buffers : they're simply
		 * offsets on gpu_complex_buffers_ data. */
		if (!gpu_pindex_buffers_.empty())
			gpu_pindex_buffers_.clear();
		std::for_each(gpu_complex_buffers_.begin(),
			gpu_complex_buffers_.end(),
			[&](cufftComplex* buf)
		{
			gpu_pindex_buffers_.push_back(buf + compute_desc_.pindex.load() * input_.get_frame_desc().frame_res());
		});

		if (!gpu_vibro_buffers_.empty())
			gpu_vibro_buffers_.clear();
		std::for_each(gpu_complex_buffers_.begin(),
			gpu_complex_buffers_.end(),
			[&](cufftComplex* buf)
		{
			gpu_vibro_buffers_.push_back(buf + compute_desc_.vibrometry_q.load() * input_.get_frame_desc().frame_res());
		});
		return (true);
	}

	void Pipeline::refresh()
	{
		ICompute::refresh();
		const camera::FrameDescriptor& input_fd = input_.get_frame_desc();
		const camera::FrameDescriptor& output_fd = output_.get_frame_desc();

		refresh_requested_ = false;
		stop_pipeline();

		if (update_n_requested_)
		{
			update_n_requested_ = false;
			update_n_parameter(compute_desc_.nsamples.load());
		}

		if (abort_construct_requested_)
		{
			std::cout << "[PIPELINE] abort_construct_requested" << std::endl;
			return;
		}

		modules_.push_back(new Module()); // C1
		modules_.push_back(new Module()); // C2
		modules_.push_back(new Module()); // F1

		if (!autofocus_requested_)
		{
			modules_[0]->push_back_worker(std::bind(
				make_contiguous_complex,
				std::ref(input_),
				std::ref(gpu_complex_buffers_[0]),
				input_length_,
				modules_[0]->stream_
				));
		}
		else
		{
			autofocus_init();
			modules_[0]->push_back_worker(std::bind(
				&Pipeline::cudaMemcpyNoReturn,
				this,
				std::ref(gpu_complex_buffers_[0]),
				af_env_.gpu_input_buffer_tmp,
				af_env_.gpu_input_size,
				cudaMemcpyDeviceToDevice
				));
		}

		if (compute_desc_.algorithm.load() == ComputeDescriptor::FFT1)
		{
			// Initialize FFT1 lens.
			if (!autofocus_requested_)
			{
				fft1_lens(
					gpu_lens_,
					input_fd,
					compute_desc_.lambda.load(),
					compute_desc_.zdistance.load(),
					static_cast<cudaStream_t>(0));
			}
			else
			{
				modules_[1]->push_back_worker(std::bind(
					fft1_lens,
					gpu_lens_,
					input_fd,
					compute_desc_.lambda.load(),
					std::ref(af_env_.z),
					modules_[1]->stream_));
			}

			// Add FFT1.
			modules_[1]->push_back_worker(std::bind(
				fft_1,
				std::ref(gpu_complex_buffers_[1]),
				gpu_lens_,
				plan1d_,
				plan2d_,
				input_fd.frame_res(),
				compute_desc_.nsamples.load(),
				compute_desc_.pindex.load(),
				compute_desc_.vibrometry_q.load(),
				modules_[1]->stream_
				));

			if (compute_desc_.vibrometry_enabled.load())
			{
				modules_[1]->push_back_worker(std::bind(
					frame_ratio,
					std::ref(gpu_pindex_buffers_[1]),
					std::ref(gpu_vibro_buffers_[1]),
					std::ref(gpu_pindex_buffers_[1]),
					input_fd.frame_res(),
					modules_[1]->stream_));
			}
		}
		else if (compute_desc_.algorithm.load() == ComputeDescriptor::FFT2)
		{
			// Initialize FFT2 lens.
			if (!autofocus_requested_)
			{
				fft2_lens(
					gpu_lens_,
					input_fd,
					compute_desc_.lambda.load(),
					compute_desc_.zdistance.load(),
					static_cast<cudaStream_t>(0));
			}
			else
			{
				modules_[1]->push_back_worker(std::bind(
					fft2_lens,
					gpu_lens_,
					input_fd,
					compute_desc_.lambda.load(),
					std::ref(af_env_.z),
					modules_[1]->stream_));
			}

			if (compute_desc_.vibrometry_enabled.load())
			{
				modules_[1]->push_back_worker(std::bind(
					fft_2,
					std::ref(gpu_pindex_buffers_[1]),
					gpu_lens_,
					plan3d_,
					plan2d_,
					input_fd.frame_res(),
					compute_desc_.nsamples.load(),
					compute_desc_.pindex.load(),
					compute_desc_.vibrometry_q.load(),
					modules_[1]->stream_
					));

				modules_[1]->push_back_worker(std::bind(
					frame_ratio,
					std::ref(gpu_pindex_buffers_[1]),
					std::ref(gpu_vibro_buffers_[1]),
					std::ref(gpu_pindex_buffers_[1]),
					input_fd.frame_res(),
					modules_[1]->stream_
					));
			}
			else
			{
				modules_[1]->push_back_worker(std::bind(
					fft_2,
					std::ref(gpu_complex_buffers_[1]),
					gpu_lens_,
					plan3d_,
					plan2d_,
					input_fd.frame_res(),
					compute_desc_.nsamples.load(),
					compute_desc_.pindex.load(),
					compute_desc_.pindex.load(),
					modules_[1]->stream_
					));
			}
		}
		else if (compute_desc_.stft_enabled.load())
		{
			// Initialize FFT1 lens.
			if (!autofocus_requested_)
			{
				fft1_lens(
					gpu_lens_,
					input_fd,
					compute_desc_.lambda.load(),
					compute_desc_.zdistance.load(),
					static_cast<cudaStream_t>(0));
			}
			else
			{
				modules_[1]->push_back_worker(std::bind(
					fft1_lens,
					gpu_lens_,
					input_fd,
					compute_desc_.lambda.load(),
					std::ref(af_env_.z),
					modules_[1]->stream_));
			}

			curr_elt_stft_ = 0;
			// Add STFT.
			/* modules_[1]->push_back_worker(std::bind(
			   stft,
			   std::ref(gpu_complex_buffers_[1]),
			   gpu_lens_,
			   gpu_stft_buffer_,
			   gpu_stft_dup_buffer_,
			   plan2d_,
			   plan1d_,
			   compute_desc_.stft_roi_zone.load(),
			   curr_elt_stft_,
			   input_fd,
			   compute_desc_.nsamples.load(),
			   compute_desc_.stft_level.load(),
			   modules_[1]->stream_
			   ));
			   */
			/* modules_[1]->push_back_worker(std::bind(
			   stft_recontruct,
			   std::ref(gpu_complex_buffers_[1]),
			   gpu_stft_dup_buffer_,
			   compute_desc_.stft_roi_zone.load(),
			   input_fd,
			   (stft_update_roi_requested_ ? compute_desc_.stft_roi_zone.load().get_width() : input_fd.width),
			   (stft_update_roi_requested_ ? compute_desc_.stft_roi_zone.load().get_height() : input_fd.height),
			   compute_desc_.pindex.load(),
			   compute_desc_.nsamples.load(),
			   modules_[1]->stream_
			   ));*/

			gpu_pindex_buffers_ = gpu_complex_buffers_;

			if (compute_desc_.vibrometry_enabled.load())
			{
				/* q frame pointer */
				cufftComplex* q = nullptr;// q_gpu_stft_buffer;

				/* modules_[1]->push_back_worker(std::bind(
				   stft_recontruct,
				   q,
				   gpu_stft_dup_buffer_,
				   compute_desc_.stft_roi_zone.load(),
				   input_fd,
				   (stft_update_roi_requested_ ? compute_desc_.stft_roi_zone.load().get_width() : input_fd.width),
				   (stft_update_roi_requested_ ? compute_desc_.stft_roi_zone.load().get_height() : input_fd.height),
				   compute_desc_.vibrometry_q.load(),
				   compute_desc_.nsamples.load(),
				   modules_[1]->stream_
				   ));*/

				modules_[1]->push_back_worker(std::bind(
					frame_ratio,
					std::ref(gpu_complex_buffers_[1]),
					q,
					std::ref(gpu_complex_buffers_[1]),
					input_fd.frame_res(),
					modules_[1]->stream_
					));
			}
			/*
				  if (average_requested_)
				  {
				  if (compute_desc_.stft_roi_zone.load().area())
				  modules_[1]->push_back_worker(std::bind(
				  &Pipeline::average_stft_caller,
				  this,
				  gpu_stft_dup_buffer_,
				  input_fd.width,
				  input_fd.height,
				  compute_desc_.stft_roi_zone.load().get_width(),
				  compute_desc_.stft_roi_zone.load().get_height(),
				  compute_desc_.signal_zone.load(),
				  compute_desc_.noise_zone.load(),
				  compute_desc_.nsamples.load(),
				  modules_[1]->stream_
				  ));
				  }*/
		}
		else
			assert(!"Impossible case.");

		/* Apply conversion to unsigned short. */
		if (compute_desc_.view_mode.load() == ComputeDescriptor::MODULUS)
		{
			modules_[0]->push_front_worker(std::bind(
				complex_to_modulus,
				std::ref(gpu_pindex_buffers_[0]),
				std::ref(gpu_float_buffers_[0]),
				input_fd.frame_res(),
				modules_[0]->stream_
				));
		}
		else if (compute_desc_.view_mode.load() == ComputeDescriptor::SQUARED_MODULUS)
		{
			modules_[0]->push_front_worker(std::bind(
				complex_to_squared_modulus,
				std::ref(gpu_pindex_buffers_[0]),
				std::ref(gpu_float_buffers_[0]),
				input_fd.frame_res(),
				modules_[0]->stream_
				));
		}
		else if (compute_desc_.view_mode.load() == ComputeDescriptor::ARGUMENT)
		{
			modules_[0]->push_front_worker(std::bind(
				complex_to_argument,
				std::ref(gpu_pindex_buffers_[0]),
				std::ref(gpu_float_buffers_[0]),
				input_fd.frame_res(),
				modules_[0]->stream_
				));
		}
		/* Note that no forms of Unwrap are supported in the Pipeline for now. */
		else
		{
			// Falling back on modulus mode.
			modules_[0]->push_front_worker(std::bind(
				complex_to_modulus,
				std::ref(gpu_pindex_buffers_[0]),
				std::ref(gpu_float_buffers_[0]),
				input_fd.frame_res(),
				modules_[0]->stream_
				));
		}

		/* [POSTPROCESSING] Everything behind this line uses output_frame_ptr */

		if (compute_desc_.shift_corners_enabled.load())
		{
			modules_[2]->push_back_worker(std::bind(
				shift_corners,
				std::ref(gpu_float_buffers_[1]),
				output_fd.width,
				output_fd.height,
				modules_[2]->stream_
				));
		}

		if (average_requested_)
		{
			Rectangle signalZone;
			Rectangle noiseZone;
			compute_desc_.signalZone(&signalZone, ComputeDescriptor::Get);
			compute_desc_.noiseZone(&noiseZone, ComputeDescriptor::Get);
			if (average_record_requested_)
			{
				modules_[2]->push_back_worker(std::bind(
					&Pipeline::average_record_caller,
					this,
					std::ref(gpu_float_buffers_[1]),
					input_fd.width,
					input_fd.height,
					signalZone,
					noiseZone,
					modules_[2]->stream_
					));

				average_record_requested_ = false;
			}
			else
			{
				modules_[2]->push_back_worker(std::bind(
					&Pipeline::average_caller,
					this,
					std::ref(gpu_float_buffers_[1]),
					input_fd.width,
					input_fd.height,
					signalZone,
					noiseZone,
					modules_[2]->stream_
					));
			}
		}

		if (compute_desc_.log_scale_enabled.load())
		{
			modules_[2]->push_back_worker(std::bind(
				apply_log10,
				std::ref(gpu_float_buffers_[1]),
				input_fd.frame_res(),
				modules_[2]->stream_
				));
		}

		if (autocontrast_requested_)
		{
			/*modules_[2]->push_back_worker(std::bind(
				autocontrast_caller,
				std::ref(gpu_float_buffers_[1]),
				input_fd.frame_res(),
				std::ref(compute_desc_),
				nullptr,
				nullptr,
				modules_[2]->stream_
				));*/

			step_count_before_refresh_ = unsigned int(modules_.size() + 1);
			request_refresh();
			autocontrast_requested_ = false;
		}

		if (compute_desc_.contrast_enabled.load())
		{
			modules_[2]->push_back_worker(std::bind(
				manual_contrast_correction,
				std::ref(gpu_float_buffers_[1]),
				input_fd.frame_res(),
				65535,
				compute_desc_.contrast_min.load(),
				compute_desc_.contrast_max.load(),
				modules_[2]->stream_
				));
		}

		if (float_output_requested_)
		{
			modules_[2]->push_back_worker(std::bind(
				&Pipeline::record_float,
				this,
				std::ref(gpu_float_buffers_[1]),
				modules_[2]->stream_
				));
		}

		if (autofocus_requested_)
		{
			modules_[2]->push_back_worker(std::bind(
				&Pipeline::autofocus_caller,
				this,
				std::ref(gpu_float_buffers_[1]),
				modules_[2]->stream_
				));
			autofocus_requested_ = false;
		}

		if (gui::InfoManager::get_manager())
			modules_[2]->push_back_worker(std::bind(
			&Pipeline::fps_count,
			this
			));

		modules_[2]->push_back_worker(std::bind(
			float_to_ushort,
			std::ref(gpu_float_buffers_[1]),
			gpu_short_buffer_,
			input_fd.frame_res(),
			modules_[2]->stream_
			));
	}

	void Pipeline::step_forward()
	{
		if (gpu_complex_buffers_.size() > 1)
		{
			std::rotate(gpu_complex_buffers_.begin(),
				gpu_complex_buffers_.begin() + 1,
				gpu_complex_buffers_.end());

			std::rotate(gpu_pindex_buffers_.begin(),
				gpu_pindex_buffers_.begin() + 1,
				gpu_pindex_buffers_.end());

			std::rotate(gpu_vibro_buffers_.begin(),
				gpu_vibro_buffers_.begin() + 1,
				gpu_vibro_buffers_.end());
		}

		if (gpu_float_buffers_.size() > 1)
		{
			std::rotate(gpu_float_buffers_.begin(),
				gpu_float_buffers_.begin() + 1,
				gpu_float_buffers_.end());
		}
	}
}