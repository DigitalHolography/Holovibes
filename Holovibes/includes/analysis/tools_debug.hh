/*! \file tools_debug.hh
 *
 * \brief Debug functions for analysis, all functions there will be deleted when no longer needed.
 */
#pragma once

float* load_CSV_to_float_array(const std::filesystem::path& path);

void load_bin_video_file(const std::filesystem::path& path, float* output, cudaStream_t stream);

void print_in_file_gpu(float* input, uint rows, uint col, std::string filename, cudaStream_t stream);

void print_in_file_cpu(float* input, uint rows, uint col, std::string filename);