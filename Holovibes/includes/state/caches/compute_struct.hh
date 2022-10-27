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
    bool divide_enabled = false;

  public:
    bool get_is_enabled() const { return is_enabled; }
    ConvolutionStruct& set_is_enabled(bool value)
    {
        is_enabled = value;
        return *this;
    }

    bool get_divide_enabled() const { return divide_enabled; }
    ConvolutionStruct& set_divide_enabled(bool value)
    {
        divide_enabled = value;
        return *this;
    }

    std::vector<float>& get_matrix_ref() { return matrix; }
    const std::vector<float>& get_matrix_ref() const { return matrix; }
};

struct ImageAccumulationStruct
{
  public:
    bool is_enabled = false;
    bool request_clear_flag = false;

  public:
    bool get_is_enabled() const { return is_enabled; }
    ConvolutionStruct& set_is_enabled(bool value)
    {
        is_enabled = value;
        return *this;
    }

    ConvolutionStruct& request_clear()
    {
        request_clear_flag = true;
        return *this;
    }
};
} // namespace holovibes
