/*! \file 
 *
 * Contains compute parameters. */
#pragma once

# include <atomic>

# include "observable.hh"
# include "geometry.hh"

namespace holovibes
{
  const static std::string version = "v2.0.0"; /*!< Current version of this project. */

  /*! \brief Contains compute parameters.
   *
   * Theses parameters will be used when the pipe is refresh.
   * It defines parameters for FFT, lens (Fresnel transforms ...),
   * post-processing (contrast, shift_corners, log scale).
   *
   * The class use the *Observer* design pattern instead of the signal
   * mechanism of Qt because classes in the namespace holovibes are
   * independent of GUI or CLI implementations. So that, the code remains
   * reusable.
   *
   * This class contains std::atomic fields to avoid concurrent access between
   * the pipe and the GUI.
   */
  struct ComputeDescriptor : public Observable
  {
    /*! \brief Select hologram methods. */
    enum fft_algorithm
    {
      FFT1,
      FFT2,
      STFT,
    };

	/*! \brief select which mode the pipe will be using*/
	enum compute_mode
	{
		DIRECT,
		HOLOGRAM,
		DEMODULATION,
	};

    /*! \brief Complex to float methods.
     *
     * Select the method to apply to transform a complex hologram frame to a
     * float frame. */
    enum complex_view_mode
    {
      MODULUS,
      SQUARED_MODULUS,
      ARGUMENT,
      UNWRAPPED_ARGUMENT,
      UNWRAPPED_ARGUMENT_2,
      UNWRAPPED_ARGUMENT_3
    };

    /*! \brief ComputeDescriptor constructor
     *
     * Initialize the compute descriptor to default values of computation. */
    ComputeDescriptor()
      : Observable()
      , algorithm(FFT1)
	  , compute_mode(DIRECT)
      , nsamples(2)
      , pindex()
      , lambda(532e-9f)
      , zdistance(1.50f)
      , view_mode(MODULUS)
      , unwrap_history_size(10)
	  , special_buffer_size(5)
      , log_scale_enabled(false)
      , shift_corners_enabled(true)
      , contrast_enabled(false)
      , vibrometry_enabled(false)
	  , convolution_enabled(false)
	  , flowgraphy_enabled(false)
      , contrast_min(1)
      , contrast_max(65535)
      , vibrometry_q()
      , autofocus_size(3)
	  , convo_matrix_width(0)
	  , convo_matrix_height(0)
	  , convo_matrix_z(0)
	  , autofocus_z_min(0.f)
	  , autofocus_z_max(1.f)
	  , autofocus_z_div(10)
	  , autofocus_z_iter(3)
	  , flowgraphy_level(3)
	  , is_cine_file(false)
    {
      pindex = 0;
      vibrometry_q = 0;
    }

    /*! \brief Assignment operator
     *
     * The assignment operator is explicitely defined because std::atomic type
     * does not allow to generate assignments operator automatically. */
    ComputeDescriptor& operator=(const ComputeDescriptor& cd);


    /*! Hologram algorithm. */
    std::atomic<enum fft_algorithm> algorithm;
	/*! Computing mode used by the pipe */
	std::atomic<enum compute_mode> compute_mode;

	/*! Convolution matrix length. */
	std::atomic<unsigned int> convo_matrix_width;
	/*! Convolution matrix height. */
	std::atomic<unsigned int> convo_matrix_height;
	/*! Convolution matrix z. */
	std::atomic<unsigned int> convo_matrix_z;
	/*! Computing mode used by the pipe */
	std::vector<float> convo_matrix;

    /*! Number of samples in which apply the fft on. */
    std::atomic<unsigned short> nsamples;
    /*! p-th output component to show. */
    std::atomic_ushort pindex;

	/*! Set Flowgraphy level: */
	std::atomic<unsigned int> flowgraphy_level;

    /*! Lambda in meters. */
    std::atomic<float> lambda;
    /*! Sensor-to-object distance. */
    std::atomic<float> zdistance;

    /*! Complex to float method. */
    std::atomic<enum complex_view_mode> view_mode;

    /*! TODO: */
    std::atomic<int> unwrap_history_size;

	/*! Special buffer size*/
	std::atomic<int> special_buffer_size;

	/*! Is convolution processing enabled. */
	std::atomic<bool> convolution_enabled;

	/*! Is convolution processing enabled. */
	std::atomic<bool> flowgraphy_enabled;

    /*! Is log scale post-processing enabled. */
    std::atomic<bool> log_scale_enabled;

    /*! Is FFT shift corners post-processing enabled. */
    std::atomic<bool> shift_corners_enabled;

    /*! Is manual contrast post-processing enabled. */
    std::atomic<bool> contrast_enabled;
    /*! Contrast minimal range value. */
    std::atomic<float> contrast_min;
    /*! Contrast maximal range value. */
    std::atomic<float> contrast_max;

    /*! Is vibrometry method enabled. */
    std::atomic<bool> vibrometry_enabled;
    /*! q-th output component of FFT to use with vibrometry. */
    std::atomic<unsigned short> vibrometry_q;
    /*! Average mode signal zone */
    std::atomic<Rectangle> signal_zone;
    /*! Average mode noise zone */
    std::atomic<Rectangle> noise_zone;

    /*! Z minimal range for autofocus. */
    std::atomic<float> autofocus_z_min;
    /*! Z maximal range for autofocus. */
    std::atomic<float> autofocus_z_max;
    /*! Number of points of autofocus between the z range. */
    std::atomic<unsigned int> autofocus_z_div;
    /*! Number of iterations of autofocus between the z range. */
    std::atomic<unsigned int> autofocus_z_iter;
    /*! Selected zone in which apply the autofocus algorithm. */
    std::atomic<Rectangle> autofocus_zone;
    /*! Height of the matrix used inside the autofocus calculus. */
    std::atomic<unsigned int> autofocus_size;

    /*! Selected zone in which apply the stft algorithm. */
    std::atomic<Rectangle> stft_roi_zone;

	/*! Is read file a .cine file. */
	std::atomic<bool> is_cine_file;
  };
}