#ifndef AVERAGE_CUH
# define AVERAGE_CUH

# include <cuda_runtime.h>
# include <device_launch_parameters.h>
# include <cmath>
# include <vector>
# include <tuple>

# include "hardware_limits.hh"
# include "geometry.hh"

void make_average_plot(
  float *input,
  const unsigned int width,
  const unsigned int height,
  std::vector<std::tuple<float, float, float>>& output,
  holovibes::Rectangle& signal,
  holovibes::Rectangle& noise);

#endif /* !AVERAGE_CUH */