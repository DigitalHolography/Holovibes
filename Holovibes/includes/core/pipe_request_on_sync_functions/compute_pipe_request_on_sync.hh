#pragma once

#include "pipe_request_on_sync.hh"
#include "compute_struct.hh"

namespace holovibes
{
class ComputePipeRequestOnSync : public PipeRequestOnSync
{
  public:
    template <typename T>
    void operator()(typename T::ConstRefType, typename T::ConstRefType, Pipe& pipe)
    {
    }

    template <>
    void operator()<BatchSize>(int new_value, int old_value, Pipe& pipe);

    template <>
    void operator()<TimeStride>(int new_value, int old_value, Pipe& pipe);

    template <>
    void operator()<TimeTransformationSize>(uint new_value, uint old_value, Pipe& pipe);

    template <>
    void operator()<Convolution>(const ConvolutionStruct& new_value, const ConvolutionStruct& old_value, Pipe& pipe);

    template <>
    void operator()<TimeTransformationCutsEnable>(bool new_value, bool old_value, Pipe& pipe);
};
} // namespace holovibes
