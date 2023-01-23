#pragma once

#include "logger.hh"

namespace holovibes
{
class DefaultGSHOnChange
{
  public:
    template <typename T>
    void operator()(typename T::ValueType&)
    {
    }

    template <typename T>
    bool change_accepted(typename T::ConstRefType)
    {
        return true;
    }
};
} // namespace holovibes
