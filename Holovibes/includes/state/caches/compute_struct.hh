/*! \file
 *
 * \brief #TODO Add a description for this file
 */

#pragma once

#include <vector>
#include <string>

#define UID_CONVOLUTION_TYPE_DEFAULT "None"

namespace holovibes
{
struct ConvolutionStruct
{
  public:
    bool is_enabled = false;
    std::vector<float> matrix = {};
    bool divide_enabled = false;
    std::string name = UID_CONVOLUTION_TYPE_DEFAULT;

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

    const std::string& get_name() const { return name; }
    ConvolutionStruct& set_name(const std::string_view value)
    {
        name = value;
        return *this;
    }
};

} // namespace holovibes
