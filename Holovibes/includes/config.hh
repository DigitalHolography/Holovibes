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
			this->input_queue_max_size = 128;
			this->output_queue_max_size = 256;
			this->float_queue_max_size = 20;
			this->flush_on_refresh = 1;
			this->frame_timeout = 100000;
			this->reader_buf_max_size = 64;
			this->unwrap_history_size = 20;
			this->import_pixel_size = 5.42f;
			this->set_cuda_device = 1;
			this->auto_device_number = 1;
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
			this->float_queue_max_size = o.float_queue_max_size;
			this->flush_on_refresh = o.flush_on_refresh;
			this->frame_timeout = o.frame_timeout;
			this->reader_buf_max_size = o.reader_buf_max_size;
			this->unwrap_history_size = o.unwrap_history_size;
			this->import_pixel_size = o.import_pixel_size;
			this->set_cuda_device = o.set_cuda_device;
			this->auto_device_number = o.auto_device_number;
			this->device_number = o.device_number;
			return (*this);
		}

		/*! \brief Max size of input queue in number of images. */
		unsigned int input_queue_max_size;
		/*! \brief Max size of output queue in number of images. */
		unsigned int output_queue_max_size;
		/*! \brief Max size of float output queue in number of images. */
		unsigned int float_queue_max_size;
		/*! \brief Flush input queue on compute::refresh */
		bool          flush_on_refresh;
		unsigned int  frame_timeout;
		/*! \brief Max number of images read each time by thread_reader. */
		unsigned int reader_buf_max_size;
		/*! Max size of unwrapping corrections in number of images.
		 *
		 * Determines how far, meaning how many iterations back, phase corrections
		 * are taken in order to be applied to the current phase image. */
		unsigned int unwrap_history_size;
		/*! \brief default import pixel size, can't be found in .raw*/
		float        import_pixel_size;
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
