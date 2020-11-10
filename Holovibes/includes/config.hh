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

/*! \file
 *
 * Store some information that need to be accessed globally. */
#pragma once

namespace holovibes
{
	/*! \brief Store some information that need to be accessed globally.
	 */
	class Config
	{
	public:
		/*! \brief Construct the config with most commonly used values.*/
		Config()
		{
			this->input_queue_max_size = 256;
			this->output_queue_max_size = 64;
			this->time_filter_cuts_output_buffer_size = 8;
			this->flush_on_refresh = false;
			this->frame_timeout = 100000;
			this->file_buffer_size = 32;
			this->unwrap_history_size = 20;
			this->set_cuda_device = true;
			this->auto_device_number = true;
			this->device_number = 0;
		}

		/*! \brief Copy constructor.*/
		Config(const Config& o)
		{
			*this = o;
		}

		/*! \brief Assignement operator.*/
		Config& operator=(const Config& o)
		{
			this->input_queue_max_size = o.input_queue_max_size;
			this->output_queue_max_size = o.output_queue_max_size;
			this->time_filter_cuts_output_buffer_size = o.time_filter_cuts_output_buffer_size;
			this->flush_on_refresh = o.flush_on_refresh;
			this->frame_timeout = o.frame_timeout;
			this->file_buffer_size = o.file_buffer_size;
			this->unwrap_history_size = o.unwrap_history_size;
			this->set_cuda_device = o.set_cuda_device;
			this->auto_device_number = o.auto_device_number;
			this->device_number = o.device_number;
			return *this;
		}

		/*! \brief Max size of input queue in number of images. */
		unsigned int input_queue_max_size;
		/*! \brief Max size of output queue in number of images. */
		unsigned int output_queue_max_size;
		/*! \brief Max size of time filter cuts queue in number of images. */
		unsigned int time_filter_cuts_output_buffer_size;
		/*! \brief Flush input queue at start of compute::exec. (When the pipe is created) */
		bool          flush_on_refresh;
		//! Obsolete. Now using the one in the camera ini file.
		unsigned int  frame_timeout;
		/*! \brief Max number of frames read each time by the thread_reader. */
		unsigned int file_buffer_size;
		/*! Max size of unwrapping corrections in number of images.
		 *
		 * Determines how far, meaning how many iterations back, phase corrections
		 * are taken in order to be applied to the current phase image. */
		unsigned int unwrap_history_size;
		/* \brief Determines if Cuda device has to be set*/
		bool		set_cuda_device;
		/* \brief Determines if Cuda device number is automaticly set*/
		bool		auto_device_number;
		/* \brief Determines if Cuda device number is set manually*/
		unsigned int device_number;
	};
}

namespace global
{
	extern holovibes::Config global_config;
}
