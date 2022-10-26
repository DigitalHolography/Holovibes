/*! \file
 *
 * \brief #TODO Add a description for this file
 */

#pragma once

#include <vector>

namespace holovibes
{
struct ConvolutionStruct
{
  public:
    bool is_enabled = false;
    std::vector<float> matrix = {};

  public:
    bool get_is_enabled() const { return is_enabled; }
    ConvolutionStruct& set_is_enabled(bool value)
    {
        is_enabled = value;
        return *this;
    }

    std::vector<float>& get_matrix_ref() { return matrix; }
    const std::vector<float>& get_matrix_ref() const { return matrix; }
};
} // namespace holovibes
