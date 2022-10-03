#pragma once

#include "pipe.hh"
#include "parameters_handler.hh"

namespace holovibes
{
class PipeFunction
{
  public:
    template <typename T>
    bool test(const T& value)
    {
        return value.get_has_been_synchronized();
    }

    template <typename T>
    void call(const T&, Pipe& pipe)
    {
    }

    template <>
    void call<BatchSize>(const BatchSize& batch_size, Pipe& pipe)
    {
    }
};
} // namespace holovibes
