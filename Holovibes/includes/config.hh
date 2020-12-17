/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

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
        this->frame_record_queue_max_size = 64;
        this->output_queue_max_size = 64;
        this->time_transformation_cuts_output_buffer_size = 8;
        this->flush_on_refresh = false;
        this->frame_timeout = 100000;
        this->file_buffer_size = 32;
        this->unwrap_history_size = 20;
        this->set_cuda_device = true;
        this->auto_device_number = true;
        this->device_number = 0;
    }

    /*! \brief Copy constructor.*/
    Config(const Config& o) { *this = o; }

    /*! \brief Assignement operator.*/
    Config& operator=(const Config& o)
    {
        this->input_queue_max_size = o.input_queue_max_size;
        this->frame_record_queue_max_size = o.frame_record_queue_max_size;
        this->output_queue_max_size = o.output_queue_max_size;
        this->time_transformation_cuts_output_buffer_size =
            o.time_transformation_cuts_output_buffer_size;
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
    /*! \brief Max size of frame record queue in number of images. */
    unsigned int frame_record_queue_max_size;
    /*! \brief Max size of output queue in number of images. */
    unsigned int output_queue_max_size;
    /*! \brief Max size of time transformation cuts queue in number of images.
     */
    unsigned int time_transformation_cuts_output_buffer_size;
    /*! \brief Flush input queue at start of compute::exec. (When the pipe is
     * created) */
    bool flush_on_refresh;
    //! Obsolete. Now using the one in the camera ini file.
    unsigned int frame_timeout;
    /*! \brief Max number of frames read each time by the thread_reader. */
    unsigned int file_buffer_size;
    /*! Max size of unwrapping corrections in number of images.
     *
     * Determines how far, meaning how many iterations back, phase corrections
     * are taken in order to be applied to the current phase image. */
    unsigned int unwrap_history_size;
    /* \brief Determines if Cuda device has to be set*/
    bool set_cuda_device;
    /* \brief Determines if Cuda device number is automaticly set*/
    bool auto_device_number;
    /* \brief Determines if Cuda device number is set manually*/
    unsigned int device_number;
};
} // namespace holovibes

namespace global
{
extern holovibes::Config global_config;
}
