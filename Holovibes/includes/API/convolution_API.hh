#pragma once

#include "API_detail.hh"

namespace holovibes::api
{

void disable_convolution();
void enable_convolution(const std::string& filename);
void load_convolution_matrix(const std::string& file);

} // namespace holovibes::api
