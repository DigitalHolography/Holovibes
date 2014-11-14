#ifndef COMPUTE_DESCRIPTOR_HH
# define COMPUTE_DESCRIPTOR_HH

namespace holovibes
{
  /*! \brief The aim of this structure is to contain CUDA compute parameters. */
  struct ComputeDescriptor
  {
    enum fft_algorithm
    {
      FFT1,
      FFT2,
    };

    enum complex_view_mode
    {
      MODULUS,
      SQUARED_MODULUS,
      ARGUMENT,
    };

    ComputeDescriptor()
      : algorithm(FFT1)
      , nsamples(0)
      , pindex(0)
      , lambda(0.0f)
      , zdistance(0.0f)
      , view_mode(SQUARED_MODULUS)
      , log_scale_enabled(false)
      , shift_corners_enabled(false)
      , contrast_enabled(false)
      , contrast_min(0)
      , contrast_max(65535)
      , vibrometry_q(0)
      , vibrometry_p(0)
    {}

    enum fft_algorithm algorithm;
    /*! Number of samples in which apply the fft on. */
    unsigned int nsamples;
    /*! p-th output component to show. */
    unsigned int pindex;
    /*! Lambda in meters. */
    float lambda;
    /*! Sensor-to-object distance. */
    float zdistance;
    enum complex_view_mode view_mode;
    bool log_scale_enabled;
    bool shift_corners_enabled;
    bool contrast_enabled;
    unsigned short contrast_min;
    unsigned short contrast_max;
    unsigned short vibrometry_q;
    unsigned short vibrometry_p;
  };
}

#endif /* !COMPUTE_DESCRIPTOR_HH */