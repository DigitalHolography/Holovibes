#ifndef HARDWARE_LIMITS_HH_
# define HARDWARE_LIMITS_HH_

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <math.h>

unsigned int get_max_threads_1d();
unsigned int get_max_threads_2d();
unsigned int get_max_blocks();

#endif /* !HARDWARE_LIMITS_HH_ */