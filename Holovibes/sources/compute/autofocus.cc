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

#include "autofocus.hh"
#include "compute_descriptor.hh"
#include "frame_desc.hh"
#include "autofocus.cuh"
#include "icompute.hh"
#include "preprocessing.cuh"

namespace holovibes
{
	namespace compute
	{
		Autofocus::Autofocus(FnVector& fn_vect,
			const CoreBuffers& buffers,
			holovibes::Queue& input,
			holovibes::ComputeDescriptor& cd,
			ICompute *Ic)
			: fn_vect_(fn_vect)
			, buffers_(buffers)
			, input_(input)
			, fd_(input.get_frame_desc())
			, cd_(cd)
			, Ic_(Ic)
		{}

		void Autofocus::insert_init()
		{
			// initializing autofocus process and refreshing again
			fn_vect_.push_back([=]() {autofocus_init(); });
			Ic_->request_refresh();
		}

		void Autofocus::insert_restore()
		{
			if (af_env_.state == af_state::RUNNING)
				fn_vect_.push_back([=]() {autofocus_restore(); });
		}

		void Autofocus::insert_autofocus()
		{
			if (af_env_.state == af_state::RUNNING)
				fn_vect_.push_back([=]() {autofocus_caller(); });
		}

		void Autofocus::insert_copy()
		{
			af_env_.stft_index--;
			// We have to compute dest outside the lambda, otherwise it will be evaluated when the lambda will be called.
			auto dest = af_env_.gpu_input_buffer_tmp.get() + af_env_.stft_index * input_.get_pixels();
			fn_vect_.push_back([=]() {make_contiguous_complex(input_, dest); });
			if (af_env_.stft_index == 0)
			{
				af_env_.state = af_state::RUNNING;
				af_env_.stft_index = af_env_.nsamples;
			}
			Ic_->request_refresh();
		}

		float Autofocus::get_zvalue() const
		{
			return af_env_.state == RUNNING ? af_env_.z : cd_.zdistance;
		}

		void Autofocus::autofocus_init()
		{
			// Autofocus needs to work on the same images. It will computes on copies.
			try
			{
				// Saving stft parameters
				af_env_.old_nsamples = cd_.nsamples;
				af_env_.old_p = cd_.pindex;
				af_env_.old_steps = cd_.stft_steps;

				// Setting new parameters for faster autofocus
				af_env_.nsamples = 2;
				af_env_.p = 1;
				cd_.nsamples = af_env_.nsamples;
				cd_.pindex = af_env_.p;
				Ic_->request_update_n(cd_.nsamples);

				// Setting the steps and the frame_counter in order to call autofocus_caller only
				// once stft_queue_ is fully updated and stft is computed
				cd_.stft_steps = cd_.nsamples;
				Ic_->set_stft_frame_counter(af_env_.nsamples);

				af_env_.stft_index = af_env_.nsamples - 1;
				af_env_.state = af_state::COPYING;

				Ic_->notify_observers();

				af_env_.gpu_frame_size = sizeof(cufftComplex) * fd_.frame_res();
				// We want to save 'nsamples' frame, in order to entirely fill the stft_queue_
				af_env_.gpu_input_size = fd_.frame_res() * cd_.nsamples;

				af_env_.gpu_input_buffer_tmp.resize(af_env_.gpu_input_size);

				// It saves only one frames in the end of gpu_input_buffer_tmp
				make_contiguous_complex(
					input_,
					af_env_.gpu_input_buffer_tmp.get() + af_env_.stft_index * fd_.frame_res());

				cd_.autofocusZone(af_env_.zone, AccessMode::Get);
				/* Compute square af zone. */
				const unsigned int zone_width = af_env_.zone.width();
				const unsigned int zone_height = af_env_.zone.height();

				af_env_.af_square_size = upper_window_size(zone_width, zone_height);

				const unsigned int af_size = af_env_.af_square_size * af_env_.af_square_size;

				af_env_.gpu_float_buffer_af_zone.resize(af_size);
				/* Initialize z_*  */
				af_env_.z_min = cd_.autofocus_z_min;
				af_env_.z_max = cd_.autofocus_z_max;

				const float z_div = static_cast<float>(cd_.autofocus_z_div);

				af_env_.z_step = (af_env_.z_max - af_env_.z_min) / z_div;

				af_env_.af_z = 0.0f;

				af_env_.z_iter = cd_.autofocus_z_iter;
				af_env_.z = af_env_.z_min;
				af_env_.focus_metric_values.clear();
			}
			catch (std::exception e)
			{
				autofocus_reset();
				std::cout << e.what() << std::endl;
			}
		}

		void Autofocus::autofocus_restore()
		{
			af_env_.stft_index--;

			cudaMemcpy(buffers_.gpu_input_buffer_,
				af_env_.gpu_input_buffer_tmp.get() + af_env_.stft_index * fd_.frame_res(),
				af_env_.gpu_frame_size,
				cudaMemcpyDeviceToDevice);

			// Resetting the stft_index just before the call of autofocus_caller
			if (af_env_.stft_index == 0)
				af_env_.stft_index = af_env_.nsamples;
		}

		void Autofocus::autofocus_caller(cudaStream_t stream)
		{
			// Since stft_frame_counter and stft_steps are resetted in the init, we cannot call autofocus_caller when the stft_queue_ is not fully updated
			if (af_env_.stft_index != af_env_.nsamples)
			{
				autofocus_reset();
				std::cout << "Autofocus: shouldn't be called there. You should report this bug." << std::endl;
				return;
			}

			// Copying the square zone into the tmp buffer
			frame_memcpy(buffers_.gpu_float_buffer_, af_env_.zone, fd_.width, af_env_.gpu_float_buffer_af_zone.get(), af_env_.af_square_size, stream);

			// Evaluating function
			const float focus_metric_value = focus_metric(af_env_.gpu_float_buffer_af_zone.get(),
				af_env_.af_square_size,
				stream,
				cd_.autofocus_size);

			if (!std::isnan(focus_metric_value))
				af_env_.focus_metric_values.push_back(focus_metric_value);

			af_env_.z += af_env_.z_step;

			// End of loop
			if (Ic_->get_autofocus_stop_request() || af_env_.z > af_env_.z_max)
			{
				// Find max z
				auto biggest = std::max_element(af_env_.focus_metric_values.begin(), af_env_.focus_metric_values.end());
				const float z_div = static_cast<float>(cd_.autofocus_z_div);

				/* Case the max has not been found. */
				if (biggest == af_env_.focus_metric_values.end())
				{
					// Restoring old stft parameters
					cd_.stft_steps = af_env_.old_steps;
					cd_.nsamples = af_env_.old_nsamples;
					cd_.pindex = af_env_.old_p;
					Ic_->request_update_n(cd_.nsamples);

					autofocus_reset();
					std::cout << "Autofocus: Couldn't find a good value for z" << std::endl;
					Ic_->request_refresh();
					return;
				}
				long long max_pos = std::distance(af_env_.focus_metric_values.begin(), biggest);

				// This is our temp max
				af_env_.af_z = af_env_.z_min + max_pos * af_env_.z_step;

				// Calculation of the new max/min, taking the old step
				af_env_.z_min = af_env_.af_z - af_env_.z_step;
				af_env_.z_max = af_env_.af_z + af_env_.z_step;

				// prepare next iter
				if (--af_env_.z_iter > 0)
				{
					af_env_.z = af_env_.z_min;
					af_env_.z_step = (af_env_.z_max - af_env_.z_min) / z_div;
					af_env_.focus_metric_values.clear();
				}
			}

			// End of autofocus, free resources and notify the new z
			if (Ic_->get_autofocus_stop_request() || af_env_.z_iter <= 0)
			{
				// Restoring old stft parameters
				cd_.stft_steps = af_env_.old_steps;
				cd_.nsamples = af_env_.old_nsamples;
				cd_.pindex = af_env_.old_p;
				Ic_->request_update_n(cd_.nsamples);

				cd_.zdistance = af_env_.af_z;
				cd_.notify_observers();

				autofocus_reset();
			}
			Ic_->request_refresh();
		}

		void Autofocus::autofocus_reset()
		{
			// if gpu_input_buffer_tmp is freed before is used by cudaMemcpyNoReturn
			af_env_.gpu_float_buffer_af_zone.reset();
			af_env_.gpu_input_buffer_tmp.reset();

			//Resetting af_env_ for next use
			af_env_.focus_metric_values.clear();
			af_env_.stft_index = 0;
			af_env_.state = af_state::STOPPED;
		}
	}
}
