#pragma once

#include "parameters_handler.hh"
#include "logger.hh"

namespace holovibes
{

class OnSynchronizeFunctions
{
  public:
    template <typename T>
    void call(T& value)
    {
        value.set_has_been_synchronized(false);
    }

    template <>
    void call<ParametersHandler>(ParametersHandler& params)
    {
        params.synchronize();
    }
};

} // namespace holovibes
