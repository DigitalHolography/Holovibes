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

    ComputeDescriptor()
      : algorithm(FFT1)
      , nsamples(0)
      , pindex(0)
      , lambda(0.0f)
      , zdistance(0.0f)
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
  };
}

#endif /* !COMPUTE_DESCRIPTOR_HH */