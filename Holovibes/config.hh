#pragma once

namespace holovibes
{
  class Config
  {
  public:
    Config()
    {
      this->flush_on_refresh = 1;
      this->frame_timeout = 1e5;
      this->input_queue_max_size = 100;
      this->output_queue_max_size = 20;
    }
    Config(Config& o)
    {
      *this = o;
    }
    Config& operator=(Config& o)
    {
      this->flush_on_refresh = o.flush_on_refresh;
      this->frame_timeout = o.frame_timeout;
      this->input_queue_max_size = o.input_queue_max_size;
      this->output_queue_max_size = o.output_queue_max_size;
      return (*this);
    }

    bool  flush_on_refresh;
    unsigned int frame_timeout;
    /*! \brief Max size of input queue in number of images. */
    unsigned int input_queue_max_size;
    /*! \brief Max size of output queue in number of images. */
    unsigned int output_queue_max_size;
  };
}

namespace Global
{
  extern holovibes::Config global_config;
}