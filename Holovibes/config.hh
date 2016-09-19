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
      this->input_queue_max_size = 100;
      this->output_queue_max_size = 20;
      this->float_queue_max_size = 20;
      this->flush_on_refresh = 1;
      this->frame_timeout = 1e5;
      this->reader_buf_max_size = 20;
      this->unwrap_history_size = 20;
      this->import_pixel_size = 5.42f;
	  this->set_cuda_device = 1;
	  this->auto_device_number = 0;
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
      this->flush_on_refresh = o.flush_on_refresh;
      this->frame_timeout = o.frame_timeout;
      this->input_queue_max_size = o.input_queue_max_size;
      this->output_queue_max_size = o.output_queue_max_size;
      this->unwrap_history_size = o.unwrap_history_size;
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
