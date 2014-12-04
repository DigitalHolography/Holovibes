#ifndef COMPUTE_DESCRIPTOR_HH
# define COMPUTE_DESCRIPTOR_HH

# include <atomic>

# include "observable.hh"
# include "geometry.hh"

namespace holovibes
{
  /*! \brief The aim of this structure is to contain CUDA compute parameters. */
  struct ComputeDescriptor : public Observable
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
      : Observable()
      , algorithm(FFT1)
      , nsamples(2)
      , pindex(0)
      , lambda(532e-9f)
      , zdistance(1.50f)
      , view_mode(MODULUS)
      , log_scale_enabled(false)
      , shift_corners_enabled(false)
      , contrast_enabled(false)
      , vibrometry_enabled(false)
      , contrast_min(1)
      , contrast_max(65535)
      , vibrometry_q(0)
    {}

    ComputeDescriptor& operator=(const ComputeDescriptor& cd);

    /*! Check if values are possible. */
    void sanity_check();

    std::atomic<enum fft_algorithm> algorithm;
    /*! Number of samples in which apply the fft on. */
    std::atomic<unsigned int> nsamples;
    /*! p-th output component to show. */
    std::atomic<unsigned int> pindex;
    /*! Lambda in meters. */
    std::atomic<float> lambda;
    /*! Sensor-to-object distance. */
    std::atomic<float> zdistance;
    std::atomic<enum complex_view_mode> view_mode;
    std::atomic<bool> log_scale_enabled;
    std::atomic<bool> shift_corners_enabled;
    std::atomic<bool> contrast_enabled;
    std::atomic<bool> vibrometry_enabled;
    std::atomic<float> contrast_min;
    std::atomic<float> contrast_max;
    std::atomic<unsigned short> vibrometry_q;

    /*! Average mode signal zone */
    Rectangle signal_zone;
    /*! Average mode noise zone */
    Rectangle noise_zone;
  };
}

#endif /* !COMPUTE_DESCRIPTOR_HH */