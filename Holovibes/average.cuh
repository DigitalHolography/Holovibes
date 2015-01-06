#ifndef AVERAGE_CUH
# define AVERAGE_CUH

# include <tuple>
# include "geometry.hh"

std::tuple<float, float, float> make_average_plot(
  float *input,
  const unsigned int width,
  const unsigned int height,
  holovibes::Rectangle& signal,
  holovibes::Rectangle& noise);

#endif /* !AVERAGE_CUH */