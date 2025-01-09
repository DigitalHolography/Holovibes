/*!
 * \file gpu_stats.hh
 *
 * \brief A few function to gather and format information regarding the GPU and its memory.
 */

#pragma once

#include <cuda_runtime.h>
#include <nvml.h>
#include <string>

namespace holovibes
{
#define RED_COLORATION_RATIO 0.9f
#define ORANGE_COLORATION_RATIO 0.7f

const std::string get_load_color(float load,
                                 float max_load,
                                 float orange_ratio = ORANGE_COLORATION_RATIO,
                                 float red_ratio = RED_COLORATION_RATIO);
inline const std::string get_percentage_color(float percentage) { return get_load_color(percentage, 100); }
int get_gpu_load(nvmlUtilization_t* gpuLoad);
std::string gpu_load();
std::string gpu_load_as_number();
std::string gpu_memory_controller_load();
std::string gpu_memory_controller_load_as_number();
std::string gpu_memory();
} // namespace holovibes