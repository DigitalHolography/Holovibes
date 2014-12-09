#ifndef AVERAGE_CUH
# define AVERAGE_CUH

# include <cuda_runtime.h>
# include <cufft.h>
# include <vector>
# include <device_launch_parameters.h>
# include <math.h>
# include "hardware_limits.hh"
# include "frame_desc.hh"
# include "geometry.hh"

void make_average_plot(
  float *input,
  const unsigned int width,
  const unsigned int height,
  std::vector<float>& output,
  holovibes::Rectangle& signal,
  holovibes::Rectangle& noise);

#endif /* !AVERAGE_CUH */