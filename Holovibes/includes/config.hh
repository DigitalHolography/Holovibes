/*! \file
 *
 * \brief Store some information that need to be accessed globally.
 */
#pragma once

namespace holovibes
{
/*! \class Config
 *
 * \brief Store some information that need to be accessed globally.
 */
class Config
{
  public:
    /*! \brief Construct the config with most commonly used values. */
    Config()
    {
        this->output_queue_max_size = 64;
        this->file_buffer_size = 32;
    }

    /*! \brief Copy constructor. */
    Config(const Config& o) { *this = o; }

    /*! \brief Assignement operator. */
    Config& operator=(const Config& o)
    {
        this->output_queue_max_size = o.output_queue_max_size;
        this->file_buffer_size = o.file_buffer_size;
        return *this;
    }

    /*! \brief Max size of output queue in number of images. */
    unsigned int output_queue_max_size;
    /*! \brief Max number of frames read each time by the thread_reader. */
    unsigned int file_buffer_size;
};
} // namespace holovibes

namespace global
{
extern holovibes::Config global_config;
}
