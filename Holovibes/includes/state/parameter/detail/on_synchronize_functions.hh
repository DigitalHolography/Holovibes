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

    template <typename ParametersHandler>
    void call_handler(ParametersHandler& params)
    {
        params.synchronize();
    }
};

} // namespace holovibes
