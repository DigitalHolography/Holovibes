#pragma once

# include <string>
# include "holovibes.hh"
# include "compute_descriptor.hh"

namespace holovibes
{
  /*! \brief The aim of this structure is to contain user
  ** parameters entered with the CLI or GUI.
  **/
  struct OptionsDescriptor
  {
  public:
    /*! \brief The constructor sets default values. */
    OptionsDescriptor()
      : recorder_n_img(0)
      , recorder_filepath("")
      , queue_size(0)
      , gl_window_width(0)
      , gl_window_height(0)
      , camera()
      , is_gui_enabled(true)
      , is_gl_window_enabled(false)
      , is_recorder_enabled(false)
      , is_compute_enabled(false)
      , is_float_output_enabled(false)
      , compute_desc()
    {}

    /*! \{ \name Parameters */
    /*! \brief Number of images to record. */
    unsigned int  recorder_n_img;
    /*! \brief File path for recorder. */
    std::string   recorder_filepath;
    /*! \brief Size of the program queue in number of images. */
    unsigned int  queue_size;
    /*! \brief GL Window width. */
    unsigned int  gl_window_width;
    /*! \brief GL Window height. */
    unsigned int  gl_window_height;
    /*! \brief Selected camera */
    Holovibes::camera_type camera;
    /*! \brief File path to import. */
    std::string		file_src;
    /*! \brief File image width. */
    unsigned int  file_image_width;
    /*! \brief File image height. */
    unsigned int  file_image_height;
    /*! \brief File image depth. */
    unsigned int  file_image_depth;
    /*! \brief File image is big endian. */
    unsigned int  file_is_big_endian;
    /*! \brief Frame imported per seconds. */
    unsigned int  fps;
    /*! \brief First frame id imported. */
    unsigned int  spanStart;
    /*! \brief Last frame id imported. */
    unsigned int  spanEnd;
    /* \} */

    /*! \{ \name Enabled features */
    bool is_gui_enabled;
    bool is_gl_window_enabled;
    bool is_recorder_enabled;
    bool is_compute_enabled;
    bool is_import_mode_enabled;
    bool is_float_output_enabled;
    /*! \} */

    ComputeDescriptor compute_desc;
  };
}