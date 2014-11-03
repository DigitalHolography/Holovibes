#ifndef OPTIONS_DESCRIPTOR_HH
# define OPTIONS_DESCRIPTOR_HH

# include <string>
# include "holovibes.hh"

namespace holovibes
{
  /*! \brief The aim of this structure is to contain user
  ** parameters entered with the CLI or GUI.
  ** This is class with private fields to check inputs integrity at
  ** the earliest step.
  **/
  struct OptionsDescriptor
  {
  public:
    /* The constructor sets default values. */
    OptionsDescriptor()
      : recorder_n_img(0)
      , recorder_filepath("")
      , queue_size(0)
      , gl_window_width(0)
      , gl_window_height(0)
      , camera()
      , is_gl_window_enabled(false)
      , is_recorder_enabled(false)
      , is_1fft_enabled(false)
      , is_2fft_enabled(false)
      , nsamples(0)
      , pindex(0)
      , lambda(0.0f)
      , zdistance(0.0f)
    {}

    /* Parameters */
    /*! Number of images to record. */
    unsigned int recorder_n_img;
    /*! File path for recorder. */
    std::string recorder_filepath;
    /*! Size of the program queue in number of images. */
    unsigned int queue_size;
    /*! GL Window width. */
    unsigned int gl_window_width;
    /*! GL Window height. */
    unsigned int gl_window_height;
    /*! Selected camera */
    Holovibes::camera_type camera;
    /* Enabled features */
    bool is_gl_window_enabled;
    bool is_recorder_enabled;
    bool is_1fft_enabled;
    bool is_2fft_enabled;
    /* Number of samples in which apply the fft on. */
    unsigned int nsamples;
    /*! p-th output component to show. */
    unsigned int pindex;
    /*! Lambda in meters. */
    float lambda;
    /*! Sensor-to-object distance. */
    float zdistance;
  };
}
#endif /* !OPTIONS_DESCRIPTOR_HH */
