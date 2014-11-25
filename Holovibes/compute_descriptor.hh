#ifndef COMPUTE_DESCRIPTOR_HH
# define COMPUTE_DESCRIPTOR_HH

# include <atomic>

# include "observable.hh"

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
      , nsamples(0)
      , pindex(0)
      , lambda(0.0f)
      , zdistance(0.0f)
      , view_mode(MODULUS)
      , log_scale_enabled(false)
      , shift_corners_enabled(false)
      , contrast_enabled(false)
      , contrast_min(0)
      , contrast_max(65535)
      , vibrometry_q(0)
      , vibrometry_p(0)
    {}

    ComputeDescriptor& operator=(const ComputeDescriptor& cd)
    {
      algorithm = cd.algorithm.load();
      nsamples = cd.nsamples.load();
      pindex = cd.pindex.load();
      lambda = cd.lambda.load();
      zdistance = cd.zdistance.load();
      view_mode = cd.view_mode.load();
      log_scale_enabled = cd.log_scale_enabled.load();
      shift_corners_enabled = cd.shift_corners_enabled.load();
      contrast_enabled = cd.contrast_enabled.load();
      contrast_min = cd.contrast_min.load();
      contrast_max = cd.contrast_max.load();
      vibrometry_p = cd.vibrometry_p.load();
      vibrometry_q = cd.vibrometry_q.load();

      return *this;
    }

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
    std::atomic<unsigned short> contrast_min;
    std::atomic<unsigned short> contrast_max;
    std::atomic<unsigned short> vibrometry_q;
    std::atomic<unsigned short> vibrometry_p;
  };
}

#endif /* !COMPUTE_DESCRIPTOR_HH */