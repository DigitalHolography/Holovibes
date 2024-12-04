/*! \file tools_analysis_debug.hh
 *
 * \brief Debug functions for analysis, all functions there will be deleted when no longer needed.
 */
#pragma once

#include <cuda_runtime.h>

float* load_CSV_to_float_array(const std::filesystem::path& path);

void load_CSV_to_float_array_gpu(float* const output,
                                 const size_t size,
                                 const std::filesystem::path& path,
                                 cudaStream_t stream);

void load_bin_video_file(const std::filesystem::path& path, float* output, cudaStream_t stream);

void print_in_file_cpu(float* input, uint rows, uint col, std::string filename);

#include "tools_analysis_debug.hxx"