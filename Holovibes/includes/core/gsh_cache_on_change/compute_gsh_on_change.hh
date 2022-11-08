#pragma once

#include "default_gsh_cache_on_change.hh"
#include "compute_cache.hh"

namespace holovibes
{
class ComputeGSHOnChange
{
  public:
    template <typename T>
    void operator()(typename T::ValueType&)
    {
    }

  public:
    template <>
    void operator()<Convolution>(ConvolutionStruct& new_value);
};
} // namespace holovibes
