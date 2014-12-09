#ifndef AVERAGE_CUH
# define AVERAGE_CUH

# include <cuda_runtime.h>
# include <device_launch_parameters.h>
# include <cmath>
# include <tuple>

# include "hardware_limits.hh"
# include "geometry.hh"

std::tuple<float, float, float> make_average_plot(
  float *input,
  const unsigned int width,
  const unsigned int height,
  holovibes::Rectangle& signal,
  holovibes::Rectangle& noise);

#endif /* !AVERAGE_CUH */