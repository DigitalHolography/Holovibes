#ifndef AVERAGE_CUH
# define AVERAGE_CUH

# include <tuple>
# include "geometry.hh"
# include <cufft.h>

std::tuple<float, float, float> make_average_plot(
  float *input,
  const unsigned int width,
  const unsigned int height,
  holovibes::Rectangle& signal,
  holovibes::Rectangle& noise);


std::tuple<float, float, float> make_average_stft_plot(
  cufftComplex*         input,
  unsigned int          width,
  unsigned int          height,
  holovibes::Rectangle&  signal_zone,
  holovibes::Rectangle&  noise_zone,
  unsigned int          pindex,
  unsigned int          nsamples);

#endif /* !AVERAGE_CUH */