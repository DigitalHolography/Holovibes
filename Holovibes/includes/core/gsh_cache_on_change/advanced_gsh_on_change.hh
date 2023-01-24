#pragma once

#include "logger.hh"
#include "advanced_cache.hh"

namespace holovibes
{
class AdvancedGSHOnChange
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

  public:
    template <>
    void operator()<InputBufferSize>(uint& new_value);
    template <>
    void operator()<OutputBufferSize>(uint& new_value);
    template <>
    void operator()<FileBufferSize>(uint& new_value);
    template <>
    bool change_accepted<RecordBufferSize>(uint new_value);
};
} // namespace holovibes
